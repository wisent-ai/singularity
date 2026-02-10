"""
Clone SaaS — Code Generation Actions

generate_plan: LLM produces a structured build plan.
generate_page: Generate a single Next.js page + components.
generate_api:  Generate a single API route + Supabase migration SQL.
"""

import json
from typing import Dict
from singularity.skills.base import SkillResult

# ── Compact scaffold templates ────────────────────────────────────────

_PKG = '{"name":"%(n)s","version":"0.1.0","private":true,"scripts":{"dev":"next dev","build":"next build","start":"next start","lint":"next lint"},"dependencies":{"next":"14.2.5","react":"^18","react-dom":"^18","@supabase/supabase-js":"^2.44.0","@supabase/ssr":"^0.4.0"},"devDependencies":{"typescript":"^5","@types/node":"^20","@types/react":"^18","@types/react-dom":"^18","postcss":"^8","tailwindcss":"^3.4.1","autoprefixer":"^10.0.1"}}'
_TSCONFIG = '{"compilerOptions":{"target":"es5","lib":["dom","dom.iterable","esnext"],"allowJs":true,"skipLibCheck":true,"strict":true,"noEmit":true,"esModuleInterop":true,"module":"esnext","moduleResolution":"bundler","resolveJsonModule":true,"isolatedModules":true,"jsx":"preserve","incremental":true,"plugins":[{"name":"next"}],"paths":{"@/*":["./src/*"]}},"include":["next-env.d.ts","**/*.ts","**/*.tsx",".next/types/**/*.ts"],"exclude":["node_modules"]}'
_TW = 'import type { Config } from "tailwindcss";\nconst config: Config = { content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"], theme: { extend: {} }, plugins: [] };\nexport default config;\n'
_POSTCSS = "module.exports = { plugins: { tailwindcss: {}, autoprefixer: {} } };\n"
_CSS = "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n"
_NEXT = "/** @type {import('next').NextConfig} */\nconst nextConfig = {};\nmodule.exports = nextConfig;\n"
_SB_CLIENT = 'import { createBrowserClient } from "@supabase/ssr";\nexport function createClient() {\n  return createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!);\n}\n'
_SB_SERVER = 'import { createServerClient, type CookieOptions } from "@supabase/ssr";\nimport { cookies } from "next/headers";\nexport function createServerSupabase() {\n  const cookieStore = cookies();\n  return createServerClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!, {\n    cookies: {\n      get(name: string) { return cookieStore.get(name)?.value; },\n      set(name: string, value: string, options: CookieOptions) { try { cookieStore.set({ name, value, ...options }); } catch (e) {} },\n      remove(name: string, options: CookieOptions) { try { cookieStore.set({ name, value: "", ...options }); } catch (e) {} },\n    },\n  });\n}\n'
_LAYOUT = 'import type { Metadata } from "next";\nimport "./globals.css";\nexport const metadata: Metadata = { title: "%(t)s", description: "%(d)s" };\nexport default function RootLayout({ children }: { children: React.ReactNode }) {\n  return <html lang="en"><body className="antialiased">{children}</body></html>;\n}\n'


def _scaffold(name: str, plan: Dict) -> Dict[str, str]:
    t = plan.get("title", name)
    d = plan.get("description", f"{name} — Next.js + Tailwind + Supabase")
    return {
        "package.json": json.dumps(json.loads(_PKG % {"n": name}), indent=2),
        "tsconfig.json": json.dumps(json.loads(_TSCONFIG), indent=2),
        "tailwind.config.ts": _TW, "postcss.config.js": _POSTCSS,
        "next.config.js": _NEXT, "src/app/globals.css": _CSS,
        "src/app/layout.tsx": _LAYOUT % {"t": t, "d": d},
        "src/lib/supabase/client.ts": _SB_CLIENT,
        "src/lib/supabase/server.ts": _SB_SERVER,
        ".gitignore": "node_modules/\n.next/\n.env*.local\n",
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _parse_files(raw: str) -> Dict[str, str]:
    """Parse '// file: path' markers into {path: content}."""
    files, path, lines = {}, None, []
    for line in raw.split("\n"):
        if line.strip().startswith("// file:"):
            if path:
                files[path] = "\n".join(lines).strip() + "\n"
            path = line.strip().replace("// file:", "").strip()
            lines = []
        else:
            lines.append(line)
    if path:
        files[path] = "\n".join(lines).strip() + "\n"
    return files


def _parse_json(raw: str) -> Dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
        raise ValueError("Could not parse JSON from LLM output")


# ── generate_plan ─────────────────────────────────────────────────────

async def generate_plan(skill, params: Dict) -> SkillResult:
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="project_name is required")

    state = skill._load_state(project_name)
    analysis = state.get("analysis")
    if not analysis:
        return SkillResult(success=False, message="No analysis found — run analyze_saas first")

    custom = params.get("customizations", "")
    prompt = (
        f"Based on this SaaS analysis, create a build plan for a Next.js 14 (App Router) clone.\n\n"
        f"Analysis:\n{json.dumps(analysis, indent=2)}\n\n"
        f"{'Customizations: ' + custom if custom else ''}\n\n"
        "Return JSON with keys: title, description, pages (array of {{name, route, description, components[]}}), "
        "api_routes (array of {{name, method, path, description}}), "
        "db_schema (array of {{table, columns[{{name, type, constraints}}]}}), "
        "components (array of {{name, description, props[]}}), "
        "design_tokens ({{primary_color, secondary_color, accent_color, font_family}}). "
        "Return ONLY valid JSON."
    )

    try:
        plan = _parse_json(await skill._generate(
            prompt, system="You are a senior full-stack architect. Return only valid JSON."))
    except Exception as e:
        return SkillResult(success=False, message=f"Plan generation failed: {str(e)}")

    scaffold = _scaffold(project_name, plan)
    state["plan"] = plan
    state["generated_files"].update(scaffold)
    state["generation_status"] = {}
    skill._save_state(project_name, state)

    return SkillResult(
        success=True,
        message=f"Plan: {len(plan.get('pages', []))} pages, {len(plan.get('api_routes', []))} APIs, scaffold created",
        data={"project_name": project_name, "plan": plan, "scaffold_files": list(scaffold.keys())})


# ── generate_page ─────────────────────────────────────────────────────

async def generate_page(skill, params: Dict) -> SkillResult:
    project_name, page_name = params.get("project_name"), params.get("page_name")
    if not project_name or not page_name:
        return SkillResult(success=False, message="project_name and page_name are required")

    state = skill._load_state(project_name)
    plan = state.get("plan")
    if not plan:
        return SkillResult(success=False, message="No plan — run generate_plan first")

    spec = next((p for p in plan.get("pages", [])
                 if (p.get("name") if isinstance(p, dict) else p) == page_name), None)
    if not spec:
        return SkillResult(success=False, message=f"Page '{page_name}' not in plan")

    prompt = (
        f"Generate Next.js 14 App Router page + components.\n\nPage: {json.dumps(spec, indent=2)}\n"
        f"Design: {json.dumps(plan.get('design_tokens', {}), indent=2)}\n"
        f"Components: {json.dumps(plan.get('components', []), indent=2)}\n"
        f"Existing: {list(state.get('generated_files', {}).keys())}\n\n"
        "Use TypeScript + Tailwind. 'use client' only where needed. "
        "Mark files with '// file: src/app/...' or '// file: src/components/...'. "
        "Polished, production-ready UI."
    )

    try:
        files = _parse_files(await skill._generate(prompt))
    except Exception as e:
        return SkillResult(success=False, message=f"Page generation failed: {str(e)}")
    if not files:
        return SkillResult(success=False, message="No parseable files returned")

    state["generated_files"].update(files)
    state["generation_status"][f"page:{page_name}"] = "done"
    skill._save_state(project_name, state)
    return SkillResult(success=True, message=f"Generated page '{page_name}': {len(files)} files",
                       data={"project_name": project_name, "page_name": page_name, "files": list(files.keys())})


# ── generate_api ──────────────────────────────────────────────────────

async def generate_api(skill, params: Dict) -> SkillResult:
    project_name, route_name = params.get("project_name"), params.get("route_name")
    if not project_name or not route_name:
        return SkillResult(success=False, message="project_name and route_name are required")

    state = skill._load_state(project_name)
    plan = state.get("plan")
    if not plan:
        return SkillResult(success=False, message="No plan — run generate_plan first")

    spec = next((r for r in plan.get("api_routes", [])
                 if (r.get("name") if isinstance(r, dict) else r) == route_name), None)
    if not spec:
        return SkillResult(success=False, message=f"Route '{route_name}' not in plan")

    prompt = (
        f"Generate Next.js 14 API route + Supabase migration.\n\nRoute: {json.dumps(spec, indent=2)}\n"
        f"DB schema: {json.dumps(plan.get('db_schema', []), indent=2)}\n\n"
        "TypeScript Route Handlers. Import createServerSupabase from @/lib/supabase/server. "
        "Mark files with '// file: src/app/api/...' or '// file: supabase/migrations/...'. "
        "Include .sql migration with RLS if new tables needed."
    )

    try:
        files = _parse_files(await skill._generate(prompt))
    except Exception as e:
        return SkillResult(success=False, message=f"API generation failed: {str(e)}")
    if not files:
        return SkillResult(success=False, message="No parseable files returned")

    state["generated_files"].update(files)
    state["generation_status"][f"api:{route_name}"] = "done"
    skill._save_state(project_name, state)
    return SkillResult(success=True, message=f"Generated API '{route_name}': {len(files)} files",
                       data={"project_name": project_name, "route_name": route_name, "files": list(files.keys())})
