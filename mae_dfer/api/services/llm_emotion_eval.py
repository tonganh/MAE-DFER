import json
import re
import time
from typing import Any

_DEFAULT_EMOTIONS = (
    "Happy",
    "Sad",
    "Neutral",
    "Angry",
    "Surprise",
    "Disgust",
    "Fear",
)


def _allowed_labels(predict_out: dict[str, Any]) -> list[str]:
    names = predict_out.get("class_names")
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    return list(_DEFAULT_EMOTIONS)


def _coerce_to_allowed(raw: Any, allowed: list[str]) -> tuple[str | None, str | None]:
    if raw is None or not isinstance(raw, str):
        return None, None
    s = raw.strip()
    if not s:
        return None, None
    if s in allowed:
        return s, None
    lower_map = {a.lower(): a for a in allowed}
    sl = s.lower().replace("_", " ")
    if sl in lower_map:
        canon = lower_map[sl]
        if canon != s:
            return canon, "case_or_spacing_normalized"
        return canon, None
    return None, "unmapped"


def _fallback_emotion(predict_out: dict[str, Any], allowed: list[str]) -> str:
    for candidate in (predict_out.get("predicted_label"), "Neutral"):
        if candidate is None:
            continue
        c, _ = _coerce_to_allowed(str(candidate), allowed)
        if c is not None:
            return c
    return allowed[0]


def _constrain_emotion_field(
    obj: dict[str, Any],
    field: str,
    allowed: list[str],
    predict_out: dict[str, Any],
) -> None:
    raw = obj.get(field)
    canon, note = _coerce_to_allowed(raw, allowed)
    if canon is None:
        if raw is not None and str(raw).strip():
            obj[f"{field}_raw"] = raw
        obj[field] = _fallback_emotion(predict_out, allowed)
        obj[f"{field}_coercion"] = "unmapped_to_class_names" if note == "unmapped" else "invalid_empty_fallback"
        return
    obj[field] = canon
    if note:
        obj[f"{field}_raw"] = raw
        obj[f"{field}_coercion"] = note


def _fusion_top_labels(predict_out: dict[str, Any]) -> tuple[Any, Any]:
    video_label = predict_out.get("predicted_label")
    w = predict_out.get("speech_emotion_whisper")
    audio_label = w.get("label") if isinstance(w, dict) else None
    return video_label, audio_label


def _parse_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {}
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    return json.loads(s)


def _fusion_prompt(
    video_final: Any,
    audio_final: Any,
    allowed: list[str],
    audio_weight: float,
    video_weight: float,
) -> tuple[str, str]:
    labels_json = json.dumps(allowed, ensure_ascii=False)
    aw = round(audio_weight, 4)
    vw = round(video_weight, 4)
    if aw > vw:
        policy = (
            "Fusion policy: audio weight "
            + str(aw)
            + ", video weight "
            + str(vw)
            + " (on conflict, favor the audio top-label signal per weights; still use calibration below)."
        )
    elif vw > aw:
        policy = (
            "Fusion policy: audio weight "
            + str(aw)
            + ", video weight "
            + str(vw)
            + " (on conflict, favor the video top-label signal per weights; still use calibration below)."
        )
    else:
        policy = (
            "Fusion policy: equal weights — audio "
            + str(aw)
            + ", video "
            + str(vw)
            + "."
        )
    system = (
        "Fuse the video top-label and audio top-label into one class from class_names (video spelling/capitalization). "
        "Map audio wording if needed (e.g. happy->Happy). Audio-only heads are often over-peaked; video is milder—"
        "use fusion_weights on disagreement. "
        "Output a single JSON object with exactly one key, \"final_emotion\", whose value is one string from class_names. "
        "No other keys. No markdown."
    )
    payload = {
        "class_names": allowed,
        "fusion_weights": {"audio": aw, "video": vw},
        "video_final_emotion": video_final,
        "audio_final_emotion": audio_final,
    }
    user = (
        "class_names:\n"
        + labels_json
        + "\nInputs:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\nReturn only: {\"final_emotion\":\"<one class_names value>\"}"
    )
    print("user", user)
    # user = (
    #     # policy
    #     + "\nclass_names:\n"
    #     + labels_json
    #     + "\nInputs:\n"
    #     + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    #     + "\nReturn only: {\"final_emotion\":\"<one class_names value>\"}"
    # )
    return system, user


def _slim_final_agent(fa: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"source": fa.get("source", "openai")}
    if "final_emotion" in fa:
        out["final_emotion"] = fa["final_emotion"]
    for k in ("final_emotion_raw", "final_emotion_coercion"):
        if k in fa:
            out[k] = fa[k]
    if "parse_error" in fa:
        out["parse_error"] = fa["parse_error"]
    if "raw" in fa:
        out["raw"] = fa["raw"]
    return out


def run_llm_emotion_evaluation(predict_out: dict[str, Any], api_key: str) -> dict[str, Any]:
    from openai import OpenAI

    from .config import llm_emotion_audio_weight, openai_eval_model

    model = openai_eval_model()
    allowed = _allowed_labels(predict_out)
    audio_w = llm_emotion_audio_weight()
    video_w = round(1.0 - audio_w, 4)
    video_final, audio_final = _fusion_top_labels(predict_out)
    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    sys_u, usr_u = _fusion_prompt(video_final, audio_final, allowed, audio_w, video_w)
    r2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_u},
            {"role": "user", "content": usr_u},
        ],
        response_format={"type": "json_object"},
    )
    raw2 = (r2.choices[0].message.content or "").strip()
    try:
        final_agent = _parse_json_object(raw2)
        final_agent["source"] = "openai"
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        final_agent = {
            "final_emotion": None,
            "source": "openai",
            "parse_error": str(e),
            "raw": raw2[:2000],
        }

    _constrain_emotion_field(final_agent, "final_emotion", allowed, predict_out)
    final_agent = _slim_final_agent(final_agent)

    elapsed = round(time.perf_counter() - t0, 4)
    return {
        "class_names": list(allowed),
        "fusion_weights": {"audio": audio_w, "video": video_w},
        "openai_model": model,
        "final_agent": final_agent,
        "elapsed_sec": elapsed,
    }
