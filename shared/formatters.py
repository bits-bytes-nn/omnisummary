import re

from .constants import ContentType
from .models import ParseResult, SummaryResult


def extract_unique_urls(urls_str: str) -> list[str]:
    if not urls_str or not urls_str.strip():
        return []

    cleaned_urls = [url.strip() for url in urls_str.split("\n") if url.strip()]
    unique_urls = []
    seen_urls = set()

    for url in cleaned_urls:
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append(url)

    return unique_urls


def format_slack_message(
    parsed_result: ParseResult,
    summary_result: SummaryResult,
    max_authors: int = 3,
) -> str:
    content = parsed_result.content
    date_str = content.published_date or None
    is_youtube = content.content_type == ContentType.YOUTUBE

    author_str: str | None = None
    if is_youtube:
        author_str = content.metadata.get("raw_authors") if content.metadata else None

    if not author_str:
        if content.affiliations:
            author_str = ", ".join(content.affiliations)
        elif content.authors:
            author_str = ", ".join(content.authors[:max_authors])

    if is_youtube and author_str:
        author_str += " 채널"

    title_link = f"<{content.source_url}|{content.title}>"
    prefix_parts = []

    if date_str:
        prefix_parts.append(f"{date_str}에")
    if author_str:
        prefix_parts.append(f"{author_str}에서")

    if prefix_parts:
        prefix = " ".join(prefix_parts)
        message = f"🗞️ {prefix} 발행한 {title_link}의 요약입니다."
    else:
        message = f"🗞️ {title_link}의 요약입니다."

    summary_text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", summary_result.summary)
    message += f"\n\n{summary_text}"

    if content.metadata and content.metadata.get("urls"):
        urls = extract_unique_urls(content.metadata["urls"])
        if urls:
            message += "\n\n📎 *참고 링크*"
            for url in urls:
                message += f"\n- <{url}>"

    return message.strip()
