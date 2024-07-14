from main import make_train_valid_dfs
from inference import get_image_embeddings, find_matches


_, valid_df = make_train_valid_dfs()
model, image_embeddings = get_image_embeddings(valid_df, "/home/ahmed/lab/CLIP-tensorflow/weights/weights.pt")

find_matches(model,
             image_embeddings,
             query="a group of people dancing in a party",
             image_filenames=valid_df['image'].values,
             n=9)