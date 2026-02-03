"""SSE client for LLM side-channel harness."""

import requests

URL = "http://127.0.0.1:8000/generate_sse"

DEFAULT_PARAMS = {
    "prompt": "Napisz krótkie streszczenie o trzech zastosowaniach energii słonecznej.",
    "max_new_tokens": 120,
    "temperature": 0.0,
    "top_p": 1.0,
}


def collect_frame_lengths(url: str = URL, params: dict | None = None) -> list[int]:
    """
    Stream SSE from endpoint and collect simulated frame lengths.

    Empty lines in SSE act as separators and are skipped.
    iter_lines returns single lines (e.g. b"data: ..."); we append "\\n\\n"
    to simulate the full SSE frame format.
    """
    params = params or DEFAULT_PARAMS
    frame_lens: list[int] = []

    response = requests.get(url, params=params, stream=True, timeout=300)
    if not response.ok:
        body = response.text
        try:
            detail = response.json().get("detail", body)
        except Exception:
            detail = body
        print("Server error response:", body[:1000] if body else "(empty)")
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {detail}",
            response=response,
        )

    with response:

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            simulated_frame = raw_line + b"\n\n"
            frame_lens.append(len(simulated_frame))

            if raw_line.startswith(b"data: [DONE]"):
                break

    return frame_lens


def main() -> None:
    """Fetch SSE stream and print frame statistics."""
    frame_lens = collect_frame_lengths()
    print("Liczba ramek:", len(frame_lens))
    print("Pierwsze 10 długości ramek (bajty):", frame_lens[:10])


if __name__ == "__main__":
    main()
