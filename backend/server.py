from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy import event
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate
from sqlalchemy import text
from pgvector.psycopg2 import register_vector
from datetime import datetime, timedelta
import re
import os

app = FastAPI()
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = ["*"] if allowed_origins_env.strip() == "*" else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
eng = sa.create_engine(DB_DSN)

@event.listens_for(eng, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)

class Q(BaseModel):
    question: str


@app.post("/api/chat")
def answer(q: Q):
    print('Received question')
    qvec = ollama_embed(EMBED_MODEL, q.question)
    with eng.begin() as cx:
        # Try to detect teams mentioned in the question
        teams = cx.execute(text("SELECT team_id, city, name FROM teams")).mappings().all()
        question_lc = q.question.lower()
        mentioned_team_ids = []
        for t in teams:
            city = str(t["city"]).lower()
            name = str(t["name"]).lower()
            if city in question_lc or name in question_lc:
                mentioned_team_ids.append(int(t["team_id"]))

        # Extract a date from question (YYYY-MM-DD or MM/DD/YYYY)
        d1 = d2 = None
        iso_match = re.search(r"(20\d{2}-\d{2}-\d{2})", question_lc)
        slash_match = re.search(r"(\d{1,2}/\d{1,2}/(20\d{2}))", question_lc)
        target_date = None
        try:
            if iso_match:
                target_date = datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
            elif slash_match:
                target_date = datetime.strptime(slash_match.group(1), "%m/%d/%Y").date()
        except Exception:
            target_date = None

        if target_date:
            d1 = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            d2 = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")

        # If we have two teams and a date, try a direct lookup and short-circuit with a precise answer
        if target_date and len(mentioned_team_ids) >= 2:
            t1, t2 = list(dict.fromkeys(mentioned_team_ids))[:2]
            direct = cx.execute(text(
                """
                SELECT gd.game_id,
                       gd.game_timestamp,
                       gd.home_team_id,
                       gd.away_team_id,
                       gd.home_points,
                       gd.away_points,
                       h.city  AS home_city,
                       h.name  AS home_name,
                       a.city  AS away_city,
                       a.name  AS away_name
                FROM game_details gd
                JOIN teams h ON h.team_id = gd.home_team_id
                JOIN teams a ON a.team_id = gd.away_team_id
                WHERE DATE(gd.game_timestamp) BETWEEN :d1 AND :d2
                  AND ((gd.home_team_id IN (:t1, :t2)) AND (gd.away_team_id IN (:t1, :t2)))
                ORDER BY ABS(EXTRACT(EPOCH FROM (CAST(gd.game_timestamp AS timestamp) - CAST(:dt AS timestamp))))
                LIMIT 1
                """
            ), {"d1": (target_date - timedelta(days=3)).strftime("%Y-%m-%d"),
                "d2": (target_date + timedelta(days=3)).strftime("%Y-%m-%d"),
                "dt": target_date.strftime("%Y-%m-%d"),
                "t1": int(t1), "t2": int(t2)}).mappings().first()
            if direct:
                r = dict(direct)
                home = f"{r['home_city']} {r['home_name']}"
                away = f"{r['away_city']} {r['away_name']}"
                score = f"{r['home_points']} - {r['away_points']}"
                winner = home if r['home_points'] > r['away_points'] else away
                reply = (
                    f"On {str(r['game_timestamp'])[:10]}, {winner} won {score} against "
                    f"{away if winner == home else home}."
                )
                evidence = [{
                    "game_id": int(r['game_id']),
                    "date": str(r['game_timestamp']),
                    "home": home,
                    "away": away,
                    "score": score,
                }]
                return {"reply": reply, "evidence": evidence}

        # If we have one team and a date, find nearest game for that team
        if target_date and len(mentioned_team_ids) == 1:
            t1 = mentioned_team_ids[0]
            direct = cx.execute(text(
                """
                SELECT gd.game_id,
                       gd.game_timestamp,
                       gd.home_team_id,
                       gd.away_team_id,
                       gd.home_points,
                       gd.away_points,
                       h.city  AS home_city,
                       h.name  AS home_name,
                       a.city  AS away_city,
                       a.name  AS away_name
                FROM game_details gd
                JOIN teams h ON h.team_id = gd.home_team_id
                JOIN teams a ON a.team_id = gd.away_team_id
                WHERE DATE(gd.game_timestamp) BETWEEN :d1 AND :d2
                  AND (gd.home_team_id = :t1 OR gd.away_team_id = :t1)
                ORDER BY ABS(EXTRACT(EPOCH FROM (CAST(gd.game_timestamp AS timestamp) - CAST(:dt AS timestamp))))
                LIMIT 1
                """
            ), {"d1": (target_date - timedelta(days=7)).strftime("%Y-%m-%d"),
                "d2": (target_date + timedelta(days=7)).strftime("%Y-%m-%d"),
                "dt": target_date.strftime("%Y-%m-%d"),
                "t1": int(t1)}).mappings().first()
            if direct:
                r = dict(direct)
                home = f"{r['home_city']} {r['home_name']}"
                away = f"{r['away_city']} {r['away_name']}"
                score = f"{r['home_points']} - {r['away_points']}"
                winner = home if r['home_points'] > r['away_points'] else away
                reply = (
                    f"Closest game on/near {target_date}: {home} vs {away}. "
                    f"{winner} won {score}."
                )
                evidence = [{
                    "game_id": int(r['game_id']),
                    "date": str(r['game_timestamp']),
                    "home": home,
                    "away": away,
                    "score": score,
                }]
                return {"reply": reply, "evidence": evidence}

        # If two teams without date, return most recent meeting
        if not target_date and len(mentioned_team_ids) >= 2:
            t1, t2 = list(dict.fromkeys(mentioned_team_ids))[:2]
            direct = cx.execute(text(
                """
                SELECT gd.game_id,
                       gd.game_timestamp,
                       gd.home_team_id,
                       gd.away_team_id,
                       gd.home_points,
                       gd.away_points,
                       h.city  AS home_city,
                       h.name  AS home_name,
                       a.city  AS away_city,
                       a.name  AS away_name
                FROM game_details gd
                JOIN teams h ON h.team_id = gd.home_team_id
                JOIN teams a ON a.team_id = gd.away_team_id
                WHERE (gd.home_team_id IN (:t1, :t2)) AND (gd.away_team_id IN (:t1, :t2))
                ORDER BY gd.game_timestamp DESC
                LIMIT 1
                """
            ), {"t1": int(t1), "t2": int(t2)}).mappings().first()
            if direct:
                r = dict(direct)
                home = f"{r['home_city']} {r['home_name']}"
                away = f"{r['away_city']} {r['away_name']}"
                score = f"{r['home_points']} - {r['away_points']}"
                winner = home if r['home_points'] > r['away_points'] else away
                reply = (
                    f"Most recent meeting: {home} vs {away} on {str(r['game_timestamp'])[:10]}. "
                    f"{winner} won {score}."
                )
                evidence = [{
                    "game_id": int(r['game_id']),
                    "date": str(r['game_timestamp']),
                    "home": home,
                    "away": away,
                    "score": score,
                }]
                return {"reply": reply, "evidence": evidence}

        # Build the retrieval query with optional filters
        base_query = [
            "SELECT gd.game_id,",
            "       gd.game_timestamp,",
            "       gd.home_team_id,",
            "       h.city  AS home_city,",
            "       h.name  AS home_name,",
            "       gd.away_team_id,",
            "       a.city  AS away_city,",
            "       a.name  AS away_name,",
            "       gd.home_points,",
            "       gd.away_points",
            "FROM game_details gd",
            "JOIN teams h ON h.team_id = gd.home_team_id",
            "JOIN teams a ON a.team_id = gd.away_team_id",
            "WHERE gd.embedding IS NOT NULL",
        ]

        params = {"q": qvec, "k": 20}

        if d1 and d2:
            base_query.append("AND DATE(gd.game_timestamp) BETWEEN :d1 AND :d2")
            params.update({"d1": d1, "d2": d2})

        # If we detected teams, filter to those teams (up to two)
        if mentioned_team_ids:
            team_ids = list(dict.fromkeys(mentioned_team_ids))[:2]
            if len(team_ids) == 1:
                team_ids = team_ids * 2
            base_query.append(
                "AND ((gd.home_team_id IN (:t1, :t2)) OR (gd.away_team_id IN (:t1, :t2)))"
            )
            params.update({"t1": team_ids[0], "t2": team_ids[1]})

        base_query.append("ORDER BY gd.embedding <-> CAST(:q AS vector) LIMIT :k")
        query_text = "\n".join(base_query)

        rows = cx.execute(text(query_text), params).fetchall()

        context = ""
        evidence = []
        for row in rows:
            r = row._asdict()
            home = f"{r['home_city']} {r['home_name']}"
            away = f"{r['away_city']} {r['away_name']}"
            score = f"{r['home_points']} - {r['away_points']}"
            evidence.append({
                "game_id": int(r['game_id']),
                "date": str(r['game_timestamp']),
                "home": home,
                "away": away,
                "score": score,
            })
            context += (
                f"On {r['game_timestamp']}, the {home} hosted the {away}. "
                f"The final score was {score}.\n"
            )

    prompt = (
        "Use only the context to answer the user's question. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n---\nQuestion: {q.question}"
    )

    reply = ollama_generate(LLM_MODEL, prompt)
    return {"reply": reply, "evidence": evidence}