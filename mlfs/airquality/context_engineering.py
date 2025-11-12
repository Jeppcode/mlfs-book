import xml.etree.ElementTree as ET
import re
import inspect
from typing import get_type_hints
import json
import datetime
import torch
import sys
import pandas as pd
from openai import OpenAI
from mlfs.airquality.air_quality_data_retrieval import (
    get_historical_data_for_date,
    get_historical_data_in_date_range,
    get_future_data_in_date_range,
    get_future_data_for_date,
)
from typing import Any, Dict, List

# -----------------------------
# Rule-based, no-LLM utilities
# -----------------------------
_WEEKDAY_TO_INT = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

def _parse_relative_keyword_to_date(keyword: str, today: datetime.date) -> datetime.date | None:
    k = keyword.lower()
    if k == "today":
        return today
    if k == "yesterday":
        return today - datetime.timedelta(days=1)
    if k == "tomorrow":
        return today + datetime.timedelta(days=1)
    return None

def _next_weekday(target_weekday: int, today: datetime.date) -> datetime.date:
    days_ahead = (target_weekday - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + datetime.timedelta(days=days_ahead)

def _extract_dates_from_text(text: str) -> List[str]:
    # Very simple extractor for YYYY-MM-DD
    pattern = r"(20\d{2}-\d{2}-\d{2})"
    return re.findall(pattern, text)

def _contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)

def _rule_based_function_selection(user_query: str) -> Dict[str, Any] | None:
    """
    Very small heuristic parser to map a user query to a function call without any LLM.
    Returns a dict: {"name": <fn_name>, "arguments": {...}} or None if not enough info.
    """
    q = user_query.strip().lower()
    today = datetime.date.today()

    # 1) Explicit dates in query
    dates = _extract_dates_from_text(q)
    if len(dates) >= 2:
        # Range detected
        date_start, date_end = dates[0], dates[1]
        is_future = _contains_any(q, ["will", "tomorrow", "next", "future"])
        fn = "get_future_data_in_date_range" if is_future else "get_historical_data_in_date_range"
        return {"name": fn, "arguments": {"date_start": date_start, "date_end": date_end}}
    if len(dates) == 1:
        d = dates[0]
        # If it "will" or similar -> future single day (use range with same day)
        is_future = _contains_any(q, ["will", "tomorrow", "next", "future"])
        if is_future:
            return {"name": "get_future_data_in_date_range", "arguments": {"date_start": d, "date_end": d}}
        # Otherwise historical single day
        return {"name": "get_historical_data_for_date", "arguments": {"date": d}}

    # 2) Relative keywords
    for kw in ["today", "yesterday", "tomorrow"]:
        if kw in q:
            target = _parse_relative_keyword_to_date(kw, today)
            if target:
                ds = target.strftime("%Y-%m-%d")
                if kw == "tomorrow":
                    return {"name": "get_future_data_in_date_range", "arguments": {"date_start": ds, "date_end": ds}}
                else:
                    return {"name": "get_historical_data_for_date", "arguments": {"date": ds}}

    # 3) Natural phrases: last week / next week / rest of the week
    if "last week" in q:
        end = today
        start = end - datetime.timedelta(days=7)
        return {"name": "get_historical_data_in_date_range", "arguments": {
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
        }}
    if "next week" in q or "rest of the week" in q:
        start = today
        end = start + datetime.timedelta(days=7)
        return {"name": "get_future_data_in_date_range", "arguments": {
            "date_start": start.strftime("%Y-%m-%d"),
            "date_end": end.strftime("%Y-%m-%d"),
        }}

    # 4) "next <weekday>"
    for token in q.split():
        t = token.strip(",.?!;:")
        if t in _WEEKDAY_TO_INT and _contains_any(q, ["next", "will"]):
            nd = _next_weekday(_WEEKDAY_TO_INT[t], today)
            ds = nd.strftime("%Y-%m-%d")
            return {"name": "get_future_data_in_date_range", "arguments": {"date_start": ds, "date_end": ds}}

    # Not enough to choose
    return None

def get_context_data_rule_based(user_query: str, feature_view, weather_fg, model_air_quality) -> str:
    """
    Rule-based, no-LLM path. Tries to infer which function to call from the user query.
    Returns a formatted string similar to the LLM path.
    """
    function = _rule_based_function_selection(user_query)
    if not function:
        return "No Function needed."

    data = invoke_function(function, feature_view, weather_fg, model_air_quality)
    if isinstance(data, str):
        return data

    if isinstance(data, pd.DataFrame):
        return f'Air Quality Measurements:\n' + '\n'.join(
            [f'Date: {row["date"]}; Air Quality: {row["pm25"]}' for _, row in data.iterrows()]
        )
    return ""


def get_type_name(t: Any) -> str:
    """Get the name of the type."""
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__


def serialize_function_to_json(func: Any) -> str:
    """Serialize a function to JSON."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": type_hints.get('return', 'void').__name__
    }

    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent=2)


def get_function_calling_prompt(user_query):
    fn = """{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}"""
    example = """{"name": "get_historical_data_in_date_range", "arguments": {"date_start": "2024-01-10", "date_end": "2024-01-14"}}"""

    prompt = f"""<|im_start|>system
You are a helpful assistant with access to the following functions:

{serialize_function_to_json(get_historical_data_for_date)}

{serialize_function_to_json(get_historical_data_in_date_range)}

{serialize_function_to_json(get_future_data_for_date)}

{serialize_function_to_json(get_future_data_in_date_range)}

###INSTRUCTIONS:
- You need to choose one function to use and retrieve paramenters for this function from the user input.
- If the user query contains 'will', and specifies a single day or date, use get_future_data_in_date_range function
- If the user query contains 'will', and specifies a range of days or dates, use get_future_data_in_date_range function.
- If the user query is for future data, but only includes a single day or date, use the get_future_data_in_date_range function,
- If the user query contains 'today' or 'yesterday', use get_historical_data_for_date function.
- If the user query contains 'tomorrow', use get_future_data_in_date_range function.
- If the user query is for historical data, and specifies a range of days or dates, use use get_historical_data_for_date function.
- If the user says a day of the week, assume the date of that day is when that day next arrives.
- Do not include feature_view and model parameters.
- Provide dates STRICTLY in the YYYY-MM-DD format.
- Generate an 'No Function needed' string if the user query does not require function calling.

IMPORTANT: Today is {datetime.date.today().strftime("%A")}, {datetime.date.today()}.

To use one of there functions respond STRICTLY with:
<onefunctioncall>
    <functioncall> {fn} </functioncall>
</onefunctioncall>

###EXAMPLES

EXAMPLE 1:
- User: Hi!
- AI Assiatant: No Function needed.

EXAMPLE 2:
- User: Is this Air Quality level good or bad?
- AI Assiatant: No Function needed.

EXAMPLE 3:
- User: When and what was the minimum air quality from 2024-01-10 till 2024-01-14?
- AI Assistant:
<onefunctioncall>
    <functioncall> {example} </functioncall>
</onefunctioncall>
<|im_end|>

<|im_start|>user
{user_query}
<|im_end|>

<|im_start|>assistant"""

    return prompt


def generate_hermes(user_query: str, model_llm, tokenizer) -> str:
    """Retrieves a function name and extracts function parameters based on the user query."""

    prompt = get_function_calling_prompt(user_query)

    tokens = tokenizer(prompt, return_tensors="pt").to(model_llm.device)
    input_size = tokens.input_ids.numel()
    with torch.inference_mode():
        generated_tokens = model_llm.generate(
            **tokens,
            use_cache=True,
            do_sample=True,
            temperature=0.2,
            top_p=1.0,
            top_k=0,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        generated_tokens.squeeze()[input_size:],
        skip_special_tokens=True,
    )


def function_calling_with_openai(user_query: str, client) -> str:
    """
    Generates a response using OpenAI's chat API.

    Args:
        user_query (str): The user's query or prompt.
        instructions (str): Instructions or context to provide to the GPT model.

    Returns:
        str: The generated response from the assistant.
    """

    instructions = get_function_calling_prompt(user_query).split('<|im_start|>user')[0]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_query},
        ]
    )

    # Extract and return the assistant's reply from the response
    if completion and completion.choices:
        last_choice = completion.choices[0]
        if last_choice.message:
            return last_choice.message.content.strip()
    return ""


def extract_function_calls(completion: str) -> List[Dict[str, Any]]:
    """Extract function calls from completion."""
    completion = completion.strip()
    pattern = r"(<onefunctioncall>(.*?)</onefunctioncall>)"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        return None

    multiplefn = match.group(1)
    root = ET.fromstring(multiplefn)
    functions = root.findall("functioncall")

    return [json.loads(fn.text) for fn in functions]


def invoke_function(function, feature_view, weather_fg, model) -> pd.DataFrame:
    """Invoke a function with given arguments."""
    # Extract function name and arguments from input_data
    function_name = function['name']
    arguments = function['arguments']

    # Using Python's getattr function to dynamically call the function by its name and passing the arguments
    function_output = getattr(sys.modules[__name__], function_name)(
        **arguments,
        feature_view=feature_view,
        weather_fg=weather_fg,
        model=model,
    )

    if type(function_output) == str:
        return function_output

    # Round the 'pm25' value to 2 decimal places
    function_output['pm25'] = function_output['pm25'].apply(round, ndigits=2)
    return function_output


def get_context_data(user_query: str, feature_view, weather_fg, model_air_quality, model_llm=None, tokenizer=None, client=None) -> str:
    """
    Retrieve context data based on user query.

    Args:
        user_query (str): The user query.
        feature_view: Feature View for data retrieval.
        model_air_quality: The air quality model.
        tokenizer: The tokenizer.

    Returns:
        str: The context data.
    """
    if client:
        # Generate a response using LLM
        completion = function_calling_with_openai(user_query, client)

    else:
        # Generate a response using LLM
        completion = generate_hermes(
            user_query,
            model_llm,
            tokenizer,
        )

    # Extract function calls from the completion
    functions = extract_function_calls(completion)

    # If function calls were found
    if functions:
        # Invoke the function with provided arguments
        data = invoke_function(functions[0], feature_view, weather_fg, model_air_quality)

        # Return formatted data as string
        if isinstance(data, pd.DataFrame):
            return f'Air Quality Measurements:\n' + '\n'.join(
                [f'Date: {row["date"]}; Air Quality: {row["pm25"]}' for _, row in data.iterrows()]
            )
        # Return message if data is not updated
        return data

    # If no function calls were found, return an empty string
    return ''
