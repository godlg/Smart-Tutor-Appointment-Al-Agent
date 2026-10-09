"""执行页面真实提交逻辑；只替换DOM/fetch边界，不依赖浏览器或网络。"""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("status", [200, 403, 409, 422])
def test_page_uses_server_session_and_can_recover_invalid_browser_binding(status):
    page = (ROOT / "web/templates/index.html").read_text(encoding="utf-8")
    handler = page.split("chatForm.onsubmit = async function(e) {", 1)[1].split("    clearBtn.onclick", 1)[0]
    script = """
const assert = require('node:assert/strict');
const status = STATUS;
const storage = new Map([['tutor_session_id','stale-session']]);
const localStorage = {getItem:k=>storage.get(k), setItem:(k,v)=>storage.set(k,v), removeItem:k=>storage.delete(k)};
const chatForm = {};
const userInput = {value:'初二数学怎么学', focus(){}};
const bubble = {innerHTML:'', textContent:''};
const sendBtn = {disabled:false,innerHTML:''};
const document = {getElementById:()=>sendBtn};
const chatBox = {querySelector:()=>bubble};
let lastBotBubble;
function addMessage() {}
function showTypingIndicator() {}
function hideTypingIndicator() {}
function updateLastBotMessage(value) {bubble.textContent=value;}
let requests=0;
async function fetch(url, options) {
 requests++;
 assert.equal(JSON.parse(options.body).session_id,'stale-session');
 return {status,ok:status===200,headers:{get:()=> 'server-session'},body:{getReader(){
  assert.equal(status,200,'rejected ownership must not be parsed as chat text');
  return {async read(){return {done:true};}};
 }}};
}
chatForm.onsubmit = async function(e) { HANDLER
;(async()=>{
 await chatForm.onsubmit({preventDefault(){}});
 assert.equal(requests,1,'never silently resend a potentially stateful request');
 assert.equal(sendBtn.disabled,false);
 if(status===200) assert.equal(storage.get('tutor_session_id'),'server-session');
 else {
  assert.equal(storage.has('tutor_session_id'),false);
  assert.equal(userInput.value,'初二数学怎么学');
  assert.match(bubble.textContent || bubble.innerHTML,/会话|身份|重发|重新发送/);
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("STATUS", str(status)).replace("HANDLER", handler)
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                            encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
