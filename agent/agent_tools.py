import os

import boto3
from strands import tool

from shared import Config, EnvVars, PdfParserType, is_running_in_aws
from tools.content_parser import HTMLParser, PdfParser, YouTubeParser
from tools.output_handler import send_slack_message as _send_slack_message
from tools.summarizer import Summarizer

from .tool_state import state_manager

config = Config.load()
profile_name = os.environ.get(EnvVars.AWS_PROFILE_NAME.value) if is_running_in_aws() else config.resources.profile_name
boto_session = (
    boto3.Session(region_name=os.environ.get(EnvVars.AWS_BEDROCK_REGION.value))
    if is_running_in_aws()
    else boto3.Session(region_name=config.resources.bedrock_region_name, profile_name=profile_name)
)


@tool
async def generate_summary(seed_message: str | None = None) -> str:
    """Generate structured summary from most recently parsed content.

    Creates an LLM-generated summary from the last parsed content (HTML, PDF, or YouTube),
    formatted for Slack delivery.

    Args:
        seed_message: Optional opening comment to start the summary with. If provided, the summary
                     will begin with this user-provided message (default: None)

    Returns:
        Success message with format: "✓ Generated summary [hash]"
        where hash is a unique identifier for referencing this summary
    """
    parsed_result = state_manager.get_parse_result()

    summarizer = Summarizer(config, boto_session)
    summary_result = await summarizer.generate_summary(parsed_result, seed_message)

    parse_hash = state_manager._last_parse_hash
    if parse_hash is not None:
        summary_hash = state_manager.store_summary_result(parse_hash, summary_result)
        return f"✓ Generated summary [{summary_hash}]"

    return "✓ Generated summary"


@tool
async def parse_html(url: str) -> str:
    """Parse HTML webpage and extract structured content.

    Fetches and parses an HTML webpage, extracting main content, title, metadata, and analyzing
    figures/images using LLM-based analysis.

    Args:
        url: Valid HTTP/HTTPS URL of the HTML webpage to parse

    Returns:
        Success message with format: "✓ Parsed HTML [hash]: 'title'"
        where hash is a unique identifier for referencing this parsed result
    """

    result = await HTMLParser(config, boto_session).parse(url)
    parse_hash = state_manager.store_parse_result(url, result)
    return f"✓ Parsed HTML [{parse_hash}]: '{result.content.title}'"


@tool
async def parse_pdf(url: str) -> str:
    """Parse PDF document and extract structured content.

    Downloads and parses a PDF document, extracting text content, analyzing figures/images,
    and extracting structured metadata using LLM-based analysis.

    Args:
        url: Valid HTTP/HTTPS URL pointing to a PDF file

    Returns:
        Success message with format: "✓ Parsed PDF [hash]: 'title'"
        where hash is a unique identifier for referencing this parsed result
    """
    result = await PdfParser(config, boto_session).parse(url, PdfParserType(config.content_parser.pdf_parser_type))
    parse_hash = state_manager.store_parse_result(url, result)
    return f"✓ Parsed PDF [{parse_hash}]: '{result.content.title}'"


@tool
async def parse_youtube(url: str) -> str:
    """Parse YouTube video and extract transcript with metadata.

    Fetches and parses a YouTube video, extracting the transcript, title, and metadata
    using LLM-based analysis.

    Args:
        url: Valid YouTube URL (e.g., youtube.com/watch?v=... or youtu.be/...)

    Returns:
        Success message with format: "✓ Parsed YouTube [hash]: 'title'"
        where hash is a unique identifier for referencing this parsed result
    """
    result = await YouTubeParser(config, boto_session).parse(url)
    parse_hash = state_manager.store_parse_result(url, result)
    return f"✓ Parsed YouTube [{parse_hash}]: '{result.content.title}'"


@tool
async def send_slack_message(enable_business_channels: bool = False) -> str:
    """Send generated summary to Slack channels.

    Posts the most recent summary to configured Slack channels. Can optionally send to
    both personal and business channels.

    Args:
        enable_business_channels: If True, sends to both personal and business channels.
                                 If False, sends only to personal channels (default: False)

    Returns:
        Success message indicating which channels received the summary, or error message
        if sending failed
    """
    if state_manager.message_sent:
        return "✓ Message already sent to Slack channels (skipping duplicate)"

    parsed_result = state_manager.get_parse_result()
    summary_result = state_manager.get_summary_result()
    enable_business_channels = enable_business_channels and config.resources.enable_business_slack_channels

    success = await _send_slack_message(parsed_result, summary_result, enable_business_channels)

    if success:
        state_manager.mark_message_sent()
        channels = "personal and business" if enable_business_channels else "personal"
        return f"✓ Sent summary to '{channels}' Slack channels"
    else:
        return "✗ Failed to send Slack message. Check token and channel configuration."
