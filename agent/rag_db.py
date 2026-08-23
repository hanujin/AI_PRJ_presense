"""
PreSense RAG 논문 DB (ChromaDB 로컬).

사용법:
  data/papers/ 에 PDF 파일을 넣으면
  PaperRAG() 초기화 시 새 파일만 자동으로 파싱·청킹·임베딩.
  이미 처리된 파일은 재처리하지 않음.
"""

from __future__ import annotations
import re
from pathlib import Path

_PAPERS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'papers'
_DB_PATH    = Path(__file__).resolve().parent.parent / 'data' / 'rag_db'

CHUNK_SIZE    = 400  # words
CHUNK_OVERLAP = 50   # words


# ── 텍스트 추출 ───────────────────────────────────────────────────────────────

def _extract(path: Path) -> str:
    import pdfplumber
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return '\n'.join(texts)


def _chunk(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, buf, buf_words = [], [], 0
    for para in paragraphs:
        words = para.split()
        if buf_words + len(words) > CHUNK_SIZE and buf:
            chunks.append(' '.join(buf))
            buf       = ' '.join(buf).split()[-CHUNK_OVERLAP:] + words
            buf_words = len(buf)
        else:
            buf.extend(words)
            buf_words += len(words)
    if buf:
        chunks.append(' '.join(buf))
    return [c for c in chunks if len(c.split()) > 20]


# ── PaperRAG ─────────────────────────────────────────────────────────────────

class PaperRAG:
    """
    ChromaDB 기반 발표 스트레스 논문 검색기.

    초기화 시 data/papers/ 의 새 파일만 자동 처리.
    이미 DB에 있는 파일은 스킵.

    사용:
        rag = PaperRAG()
        papers = rag.retrieve("voice instability anxiety", k=3)
    """

    def __init__(self, db_path: str | Path | None = None,
                 papers_dir: str | Path | None = None):
        import chromadb

        _DB_PATH.mkdir(parents=True, exist_ok=True)
        self._client     = chromadb.PersistentClient(path=str(db_path or _DB_PATH))
        self._collection = self._client.get_or_create_collection(
            name='stress_papers',
            metadata={'hnsw:space': 'cosine'},
        )

        self._sync(Path(papers_dir) if papers_dir else _PAPERS_DIR)

    def _sync(self, papers_dir: Path):
        """papers_dir 의 파일 중 DB에 없는 것만 처리."""
        if not papers_dir.exists():
            print(f"⚠️  data/papers/ 없음 — RAG DB 비어 있음")
            return

        files = list(papers_dir.glob('*.pdf'))
        if not files:
            print(f"⚠️  data/papers/ 에 논문 없음")
            return

        # 이미 처리된 파일 확인 (첫 번째 청크 ID 존재 여부로 판단)
        existing = set(
            m['source']
            for m in (self._collection.get(include=['metadatas'])['metadatas'] or [])
        )

        new_files = [f for f in files if f.name not in existing]
        if not new_files:
            print(f"✅ RAG DB 최신 상태 ({self._collection.count()}개 청크, {len(files)}편)")
            return

        print(f"📄 새 논문 {len(new_files)}편 추가 중...")
        for path in new_files:
            try:
                raw    = _extract(path)
                chunks = _chunk(raw)
                if not chunks:
                    print(f"  ⚠️  {path.name} — 텍스트 추출 실패, 스킵")
                    continue
                self._collection.add(
                    ids       = [f"{path.stem}__{i}" for i in range(len(chunks))],
                    documents = chunks,
                    metadatas = [{'source': path.name, 'chunk': i}
                                 for i in range(len(chunks))],
                )
                print(f"  ✅ {path.name} ({len(chunks)}개 청크)")
            except Exception as e:
                print(f"  ⚠️  {path.name} 처리 실패: {e}")

        print(f"RAG DB: 총 {self._collection.count()}개 청크")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """쿼리와 유사한 논문 발췌 k개 반환."""
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )
        return results['documents'][0] if results['documents'] else []

