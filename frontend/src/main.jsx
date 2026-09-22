import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const examples = ['Why was Barclays fined in 2025?','What are the most common issues banks were fined for?','Compare Barclays fines in 2024 and 2025.','Which firm received the largest fine in 2025?'];

function formatDuration(value){
 if(typeof value !== 'number' || !Number.isFinite(value)) return null;
 return `${Math.round(value)} ms`;
}

function App(){
 const [q,setQ]=useState(''); const [messages,setMessages]=useState([]); const [loading,setLoading]=useState(false);
 async function ask(question=q){ if(!question.trim()||loading)return; setMessages(m=>[...m,{role:'user',text:question}]); setQ(''); setLoading(true);
  const clientStarted=performance.now();
  try{const r=await fetch(`${API}/api/ask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})}); const data=await r.json(); if(!r.ok)throw new Error(data.detail||'Request failed'); setMessages(m=>[...m,{role:'assistant',text:data.answer,sources:data.sources||[],request_duration_ms:data.request_duration_ms,response_duration_ms:data.response_duration_ms,client_duration_ms:performance.now()-clientStarted}]);}
  catch(e){setMessages(m=>[...m,{role:'assistant',text:'Sorry, I could not answer that request. Please check that the backend is running and indexed.'}] )} finally{setLoading(false)}
 }
 return <main><header><div><div className="eyebrow">FCA ENFORCEMENT RESEARCH</div><h1>Fines RAG Assistant</h1><p>Ask questions about FCA fines and Final Notices from 2024–2026.</p></div><div className="badge">RAG · Sources included</div></header>
 <section className="examples">{examples.map(x=><button key={x} onClick={()=>ask(x)}>{x}</button>)}</section>
 <section className="chat">{messages.length===0&&<div className="empty"><h2>What would you like to know?</h2><p>Try a company, year, reason, comparison or trend question.</p></div>}{messages.map((m,i)=><div key={i} className={`msg ${m.role}`}><div className="bubble">{m.text}</div>{(typeof m.request_duration_ms === 'number' || typeof m.response_duration_ms === 'number' || typeof m.client_duration_ms === 'number')&&<div className="meta">{typeof m.request_duration_ms === 'number'&&<span>Backend: {formatDuration(m.request_duration_ms)}</span>}{typeof m.response_duration_ms === 'number'&&<span>Generation: {formatDuration(m.response_duration_ms)}</span>}{typeof m.client_duration_ms === 'number'&&<span>Total: {formatDuration(m.client_duration_ms)}</span>}</div>}{m.sources?.length>0&&<div className="sources"><b>Sources</b>{m.sources.map((s,j)=><a key={j} href={s.url} target="_blank">{s.firm||s.title} · {s.year} · p.{s.page||'—'}</a>)}</div>}</div>)}{loading&&<div className="msg assistant"><div className="bubble">Searching FCA documents…</div></div>}</section>
 <form onSubmit={e=>{e.preventDefault();ask()}}><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask about an FCA fine…"/><button>Ask</button></form>
 <footer>Grounded only in the indexed FCA documents. <a href="https://www.fca.org.uk/news/news-stories/2024-fines" target="_blank">FCA 2024</a> · <a href="https://www.fca.org.uk/news/news-stories/2025-fines" target="_blank">FCA 2025</a> · <a href="https://www.fca.org.uk/news/news-stories/2026-fines" target="_blank">FCA 2026</a></footer>
 </main>
}
createRoot(document.getElementById('root')).render(<App/>);
