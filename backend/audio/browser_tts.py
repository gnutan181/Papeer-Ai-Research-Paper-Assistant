"""Safe Web Speech API component generation; answer text never becomes JS source."""

from __future__ import annotations

import json
import re


def markdown_to_speech(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[(?:S\d+|[^\]]+, p\. ?\d+(?:, (?:Figure|Table) \d+)?)\]", "", text)
    text = re.sub(r"[`*_#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tts_component_html(answer: str) -> str:
    payload = json.dumps(markdown_to_speech(answer))
    return f"""<div><label>Voice <select id="papeer-voice"></select></label>
<label>Rate <input id="papeer-rate" type="range" min="0.7" max="1.3" step="0.1" value="1"></label>
<button onclick="papeerSpeak()">Read answer</button><button onclick="speechSynthesis.cancel()">Stop</button>
<span id="papeer-tts-status" aria-live="polite"></span></div><script>
const papeerText={payload}; const select=document.getElementById('papeer-voice');
function voices(){{select.replaceChildren(...speechSynthesis.getVoices().map((v,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=v.name;return o;}}));}}
function papeerSpeak(){{if(!('speechSynthesis' in window)){{document.getElementById('papeer-tts-status').textContent='Speech synthesis is unavailable in this browser.';return;}} speechSynthesis.cancel(); const u=new SpeechSynthesisUtterance(papeerText); u.voice=speechSynthesis.getVoices()[select.value]||null; u.rate=Number(document.getElementById('papeer-rate').value); speechSynthesis.speak(u);}}
if('speechSynthesis' in window){{voices();speechSynthesis.onvoiceschanged=voices;}} else {{document.getElementById('papeer-tts-status').textContent='Speech synthesis is unavailable in this browser.';}}
</script>"""
