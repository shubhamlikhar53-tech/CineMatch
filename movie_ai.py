import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hugging Face model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading CineMatch AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

print("CineMatch AI model loaded successfully!")


def explain_movie(
    title,
    overview,
    genres="Unknown",
    year="Unknown",
    rating="Unknown",
    question=None
):
    """
    Generate a movie explanation using the local Hugging Face model.
    No external inference API is used.
    """

    if not overview or not str(overview).strip():
        overview = "No movie synopsis is available in the dataset."

    if question is None:
        question = (
            "Explain this movie, its genre, mood, main theme, "
            "and the type of viewer who might enjoy it."
        )

    system_message = """
You are CineMatch AI, a movie explanation assistant.

Rules:
1. Use the supplied movie information as your primary source.
2. Do not invent plot details, characters, endings, or cast members.
3. Clearly distinguish supplied facts from your interpretation.
4. If information is missing, say it is unavailable.
5. Explain the movie in simple, natural English.
6. Keep the answer concise and helpful.
7. Never claim you watched the movie.
"""

    user_message = f"""
Movie title: {title}
Release year: {year}
Genres: {genres}
Rating: {rating}
Movie overview: {overview}

User's question: {question}

Explain the movie using the information above.
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


if __name__ == "__main__":
    answer = explain_movie(
        title="Interstellar",
        overview=(
            "A team of explorers travels beyond this galaxy "
            "to discover whether mankind has a future among the stars."
        ),
        genres="Science Fiction, Adventure, Drama",
        year="2014",
        rating="8.7",
        question="Explain the movie's genre, mood, and central theme."
    )

    print("\nCINEMATCH AI EXPLANATION")
    print("-" * 40)
    print(answer)