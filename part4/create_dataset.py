import pandas as pd
import sqlalchemy as sa
from backend.config import DB_DSN
import json

def get_game_details():
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        df = pd.read_sql("SELECT * FROM game_details", cx)
    return df

def generate_qa_pairs(df):
    qa_pairs = []
    for _, row in df.head(20).iterrows():
        context = (
            f"On {row['game_timestamp']}, the {row['home_team_id']} hosted the {row['away_team_id']}. "
            f"The final score was {row['home_points']} to {row['away_points']}."
        )
        
        question = f"What was the score of the game between {row['home_team_id']} and {row['away_team_id']} on {row['game_timestamp']}?"
        qa_pairs.append({"question": question, "context": context})

        question = f"Who won the game between {row['home_team_id']} and {row['away_team_id']} on {row['game_timestamp']}?"
        winner = row['home_team_id'] if row['home_points'] > row['away_points'] else row['away_team_id']
        context_winner = f"{winner} won the game."
        qa_pairs.append({"question": question, "context": context_winner})

    return qa_pairs

if __name__ == "__main__":
    game_details_df = get_game_details()
    qa_dataset = generate_qa_pairs(game_details_df)
    
    with open("/app/part4/dataset.json", "w") as f:
        json.dump(qa_dataset, f, indent=2)
        
    print("Dataset created successfully at part4/dataset.json")
