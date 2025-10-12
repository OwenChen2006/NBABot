import pandas as pd
import sqlalchemy as sa
from sqlalchemy import event, text
from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed
from pgvector.psycopg2 import register_vector
from sqlalchemy.orm import sessionmaker

def main():
    print('Starting Embedding')
    eng = sa.create_engine(DB_DSN)

    @event.listens_for(eng, "connect")
    def connect(dbapi_connection, connection_record):
        register_vector(dbapi_connection)

    # Add embedding column
    with eng.begin() as cx:
        print("Adding embedding column")
        cx.execute(text("ALTER TABLE game_details ADD COLUMN IF NOT EXISTS embedding vector(768)"))

    Session = sessionmaker(bind=eng)
    session = Session()

    print("Fetching game details to embed")
    game_details = session.execute(text("SELECT game_id, game_timestamp, home_team_id, away_team_id, home_points, away_points FROM game_details WHERE embedding IS NULL")).fetchall()

    print(f"Found {len(game_details)} game details to embed")
    
    for game in game_details:
        game_dict = game._asdict()
        text_to_embed = (
            f"On {game_dict['game_timestamp']}, the {game_dict['home_team_id']} "
            f"hosted the {game_dict['away_team_id']}. The final score was "
            f"{game_dict['home_points']} to {game_dict['away_points']}."
        )
        
        embedding = ollama_embed(EMBED_MODEL, text_to_embed)

        session.execute(
            text("UPDATE game_details SET embedding = :embedding WHERE game_id = :game_id"),
            {'embedding': embedding, 'game_id': game_dict['game_id']}
        )
    
    print("Committing embeddings")
    session.commit()
    session.close()
    print('Finished Embedding')

if __name__ == "__main__":
    main()
