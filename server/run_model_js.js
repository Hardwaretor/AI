#!/usr/bin/env node
const { spawn } = require('child_process')

function readPrompt(){
  const argv = process.argv.slice(2)
  if(argv.length > 0) return argv.join(' ')
  return new Promise((resolve)=>{
    let data = ''
    process.stdin.setEncoding('utf8')
    process.stdin.on('data', chunk => data += chunk)
    process.stdin.on('end', ()=> resolve(data.trim()))
    // if no stdin after short timeout, resolve empty
    setTimeout(()=>{ if(!data) resolve('') }, 200)
  })
}

async function main(){
  const prompt = await readPrompt()
  if(!prompt){ console.log(JSON.stringify({ reply: 'No prompt provided', actions: [] })) ; return }

  // Determine binary to run: prefer GPT4ALL_BIN, fallback to MODEL_CMD
  const bin = process.env.GPT4ALL_BIN || process.env.MODEL_CMD || null
  const argsEnv = process.env.MODEL_ARGS ? process.env.MODEL_ARGS.split(' ') : []

  if(bin){
    try{
      // Build args and replace {PROMPT} placeholder if present
      let args = [].concat(argsEnv || [])
      const hasPlaceholder = args.some(a => a.includes('{PROMPT}'))
      if(hasPlaceholder){ args = args.map(a => a.replace('{PROMPT}', prompt)) }
      else { args.push(prompt) }

      const child = spawn(bin, args, { shell: false })
      let stdout = ''
      let stderr = ''
      child.stdout.on('data', d => stdout += d.toString())
      child.stderr.on('data', d => stderr += d.toString())
      child.on('close', code => {
        if(code !== 0 && !stdout){
          // return error-like JSON
          console.log(JSON.stringify({ ok:false, message: 'Model process failed', stderr }))
          return
        }
        // try to extract JSON
        try{
          const jstart = stdout.indexOf('{')
          const jend = stdout.lastIndexOf('}')
          if(jstart !== -1 && jend !== -1 && jend > jstart){
            const jsonText = stdout.slice(jstart, jend+1)
            const parsed = JSON.parse(jsonText)
            console.log(JSON.stringify(Object.assign({ ok:true }, parsed)))
            return
          }
        }catch(e){}
        // fallback: wrap output
        console.log(JSON.stringify({ ok:true, reply: stdout || ('(no output, stderr: '+stderr+')'), actions: [] }))
      })
      // write to stdin if binary expects stdin
      try{ if(child.stdin){ child.stdin.write(prompt); child.stdin.end() } }catch(e){}
      return
    }catch(err){ console.error('run_model_js spawn failed', err); console.log(JSON.stringify({ ok:false, message: String(err) })) ; return }
  }

  // fallback simulated response
  let reply = 'Simulated response: no model binary configured.'
  let actions = []
  const low = prompt.toLowerCase()
  if(/create|draw|make|box|cube|sphere|cylinder/.test(low)){
    reply = 'Simulated: plan para crear geometría incluido.'
    actions = [{ type: 'create', command: 'create box size 1x1x1 at 0 0 0' }]
  }
  console.log(JSON.stringify({ ok:true, reply, actions }))
}

main()
