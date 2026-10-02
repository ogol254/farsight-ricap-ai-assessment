"""Persistent repository; SQLite for local development, PostgreSQL in deployment.

Every user-generated record belongs to a signed demo session. No global history
endpoint exposes another reviewer's photos or submissions.
"""
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, String, create_engine, select, func, case, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Record(Base):
    __tablename__ = "ricap_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class Document(Base):
    __tablename__ = "ricap_documents"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    section: Mapped[str] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(2))
    audience: Mapped[str] = mapped_column(String(16))
    text: Mapped[str]
    effective_date: Mapped[str] = mapped_column(String(10))
    superseded: Mapped[bool] = mapped_column(default=False)


class Taxpayer(Base):
    __tablename__ = "taxpayers"
    taxpayer_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    sector: Mapped[str] = mapped_column(String(40))
    region: Mapped[str] = mapped_column(String(40))
    business_size: Mapped[str] = mapped_column(String(16))
    turnover_growth_yoy: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")


class Filing(Base):
    __tablename__ = "filings"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    taxpayer_id: Mapped[str] = mapped_column(ForeignKey("taxpayers.taxpayer_id"), index=True)
    due_date: Mapped[date] = mapped_column(Date)
    filed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    tax_due: Mapped[float] = mapped_column(Float)


class Payment(Base):
    __tablename__ = "payments"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    taxpayer_id: Mapped[str] = mapped_column(ForeignKey("taxpayers.taxpayer_id"), index=True)
    paid_at: Mapped[date] = mapped_column(Date)
    amount: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="SUCCESS")


class Repository:
    def __init__(self, url=None):
        url = url or os.getenv("DATABASE_URL", "sqlite:///artifacts/ricap.db")
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        self.persistent_remote = url.startswith("postgresql")
        if not self.persistent_remote:
            Path("artifacts").mkdir(exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True, **({"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_size": 3, "max_overflow": 2}))
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def ping(self):
        with self.session() as db:
            db.execute(select(1))

    def count(self, kind=None):
        with self.session() as db:
            query = select(func.count()).select_from(Record)
            if kind:
                query = query.where(Record.kind == kind)
            return db.scalar(query)

    def save(self, owner, kind, payload):
        with self.session.begin() as db:
            record = Record(id=str(uuid4()), owner=owner, kind=kind, payload=payload)
            db.add(record)
        return record.id

    def history(self, owner, kind=None, limit=30):
        with self.session() as db:
            query = select(Record).where(Record.owner == owner)
            if kind:
                query = query.where(Record.kind == kind)
            records = db.scalars(query.order_by(Record.created_at.desc()).limit(limit))
            return [{"id": r.id, "kind": r.kind, "created_at": r.created_at.isoformat(), "result": r.payload} for r in records]

    def get(self, owner, record_id):
        with self.session() as db:
            return db.scalar(select(Record).where(Record.id == record_id, Record.owner == owner))

    def documents(self, audience="PUBLIC", language=None):
        with self.session() as db:
            q = select(Document).where(Document.superseded.is_(False), Document.effective_date <= datetime.now(timezone.utc).date().isoformat())
            if audience != "OFFICER":
                q = q.where(Document.audience == "PUBLIC")
            if language:
                q = q.where(Document.language == language)
            return [{k: getattr(d, k) for k in ("id", "title", "section", "language", "audience", "text", "effective_date")} for d in db.scalars(q)]

    def seed_documents(self, documents):
        with self.session.begin() as db:
            for item in documents:
                if db.get(Document, item["id"]) is None:
                    db.add(Document(**item))

    def seed_revenue(self):
        """Idempotent fictional source records for the F→G scoring path."""
        with self.session.begin() as db:
            for i in range(20):
                tid = f"DEMO-{i + 1:03}"
                if db.get(Taxpayer, tid):
                    continue
                db.add(Taxpayer(taxpayer_id=tid, sector=["RETAIL", "SERVICES", "MANUFACTURING"][i % 3],
                               region="DEMO-NORTH", business_size="SMALL", turnover_growth_yoy=-.01 * i))
                db.flush()
                for month in range(1, 13):
                    due = date(2025, month, 15)
                    db.add(Filing(id=f"{tid}-f-{month}", taxpayer_id=tid, due_date=due,
                                  filed_date=due + timedelta(days=10 if month <= i % 12 else -2), tax_due=1000))
                    db.add(Payment(id=f"{tid}-p-{month}", taxpayer_id=tid, paid_at=due,
                                   amount=1000 * (1 - i / 25), status="SUCCESS"))

    def revenue_features(self, cutoff=date(2026, 1, 1)):
        """Pre-aggregate independently to avoid multiplying payments by filings.

        The fixed synthetic snapshot is immutable and explicitly dated. This is
        not a reconstruction of historical mutable production source tables.
        """
        start12 = date(cutoff.year - 1, cutoff.month, 1)
        start24 = date(cutoff.year - 2, cutoff.month, 1)
        ff = (select(Filing.taxpayer_id,
                     func.sum(case((Filing.due_date >= start24, 1), else_=0)).label("filings"),
                     func.sum(case(((Filing.due_date >= start24) & or_(Filing.filed_date.is_(None), Filing.filed_date > Filing.due_date), 1), else_=0)).label("late"),
                     func.sum(case((Filing.due_date >= start12, Filing.tax_due), else_=0)).label("due"))
              .where(Filing.due_date < cutoff).group_by(Filing.taxpayer_id).subquery())
        pf = (select(Payment.taxpayer_id,
                     func.sum(case((Payment.paid_at >= start12, Payment.amount), else_=0)).label("paid"),
                     func.max(Payment.paid_at).label("last_paid"))
              .where(Payment.paid_at < cutoff, Payment.status == "SUCCESS").group_by(Payment.taxpayer_id).subquery())
        query = select(Taxpayer, ff.c.filings, ff.c.late, ff.c.due, pf.c.paid, pf.c.last_paid).outerjoin(ff, ff.c.taxpayer_id == Taxpayer.taxpayer_id).outerjoin(pf, pf.c.taxpayer_id == Taxpayer.taxpayer_id).where(Taxpayer.status == "ACTIVE").order_by(Taxpayer.taxpayer_id)
        with self.session() as db:
            return [{"taxpayer_id": t.taxpayer_id, "sector": t.sector, "region": t.region, "business_size": t.business_size,
                     "turnover_growth_yoy": t.turnover_growth_yoy, "filings_24m": int(filings or 0), "late_filings_24m": int(late or 0),
                     "payment_ratio_12m": (float(paid or 0) / float(due)) if due else None,
                     "days_since_last_payment": (cutoff - last).days if last else None}
                    for t, filings, late, due, paid, last in db.execute(query)]

    def close(self):
        self.engine.dispose()
