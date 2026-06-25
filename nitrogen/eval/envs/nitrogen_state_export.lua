-- nitrogen_state_export.lua — Solarus ground-truth state export hook.
--
-- Appended to the local ZSDX quest's data/main.lua by the env at boot (idempotent, marker-guarded —
-- see solarus_zelda.py _ensure_state_overlay). It must go AFTER the quest's own
-- `function sol.main:on_update`, so it can't be injected via the engine's `-s=` flag (that runs before
-- main.lua and gets clobbered). The committed repo + the ZSDX source stay stock; the env applies this
-- overlay to the locally-built (gitignored) quest so a fresh build "just works" with no manual patch.
--
-- Registers an on_update event that writes hero position / life / map / menu state to the file named by
-- the SOLARUS_STATE_EXPORT env var (falls back to a quest-write-dir file). Self-contained and tolerant
-- of a sandboxed Lua std lib (io/os may be absent).

local path = nil
if os ~= nil and type(os.getenv) == "function" then
  local ok, v = pcall(os.getenv, "SOLARUS_STATE_EXPORT")
  if ok then path = v end
end

local function make_writer(p)
  if p ~= nil and p ~= "" and io ~= nil and type(io.open) == "function" then
    local tmp = p .. ".tmp"
    return function(line)
      local f = io.open(tmp, "w")
      if f then
        f:write(line); f:flush(); f:close()
        if os ~= nil and type(os.rename) == "function" and pcall(os.rename, tmp, p) then return end
        local g = io.open(p, "w")
        if g then g:write(line); g:flush(); g:close() end
      end
    end
  end
  if sol.file ~= nil and type(sol.file.open) == "function" then
    return function(line)
      local f = sol.file.open(p or "nitrogen_solarus_state.txt", "w")
      if f then f:write(line); f:flush(); f:close() end
    end
  end
  return function(line) print("NITROGEN_SOLARUS_STATE " .. line) end
end

local write_state = make_writer(path)

local function clean(v) return tostring(v or "none"):gsub("%s+", "_") end
local function b2i(v) return v and 1 or 0 end

local function export_state()
  local running, x, y, layer = 0, 0, 0, 0
  local life, max_life, paused, in_menu, map_id = 0, 0, 0, 1, "menu"
  local game = sol.main.get_game()
  if game ~= nil then
    local okp, pv = pcall(function() return type(game.is_paused) == "function" and game:is_paused() end)
    if okp then paused = b2i(pv) end
    local okh, hero = pcall(function() return game:get_hero() end)
    local okm, map = pcall(function() return game:get_map() end)
    if okh and hero ~= nil and okm and map ~= nil then
      local okpos, px, py, pl = pcall(function() return hero:get_position() end)
      if okpos then
        running, in_menu = 1, paused
        x, y, layer = tonumber(px) or 0, tonumber(py) or 0, tonumber(pl) or 0
        local okl, l = pcall(function() return game:get_life() end); if okl then life = tonumber(l) or 0 end
        local okml, ml = pcall(function() return game:get_max_life() end); if okml then max_life = tonumber(ml) or 0 end
        local okid, id = pcall(function() return map:get_id() end); if okid then map_id = clean(id) end
      end
    end
  end
  write_state(string.format("%d %.2f %.2f %d %d %d %d %d %s\n",
      running, x, y, layer, life, max_life, paused, in_menu, map_id))
end

sol.main:register_event("on_update", export_state)
