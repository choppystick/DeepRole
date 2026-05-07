"""
Avalon solo-test GUI.

Plays a single 5-player Avalon game where the user controls every player.
Visual layout (humanoid figures arranged in a circle) shows each player's
"open role" — the role visible to the user given their own hidden role.
Hover a humanoid to see that player's hidden role (debug aid).

Phase actions are entered as a single line in the entry box at the bottom.
Default shorthand: list 1-indexed player numbers; unlisted players take the
opposite action.

  Propose      "1 3"            (players 1 and 3 go on the mission)
  Vote         "1 3 5"          (P1,P3,P5 vote YES; P2,P4 NO)
  Mission      "1"              (P1 succeeds; the rest of the team fails)
  Assassinate  "3"              (target player number)

Explicit form is also accepted if you include words:
  Vote         "1. Y 2. N 3. Y 4. N 5. Y"
  Mission      "1. S 3. F"

Run:  python -m src.avalon_gui
"""
from __future__ import annotations

import math
import random
import re
import tkinter as tk
from tkinter import ttk

import numpy as np

from src.game import (
    new_game, GameState, Phase, Role,
    NUM_PLAYERS, TEAM_SIZES, MAX_PROPOSALS, NUM_ROUNDS,
)
from src.assignments import ASSIGNMENTS, evil_indices
from src.beliefs import BeliefTracker
from src.consistency import consistency_mask


USER_PLAYER = 0  # the human always plays as Player 1 (0-indexed internally)

# Catppuccin-ish palette
BG       = "#1e1e2e"
PANEL    = "#181825"
FG       = "#cdd6f4"
ACCENT   = "#f5c2e7"
GOOD     = "#a6e3a1"
EVIL     = "#f38ba8"
NEUTRAL  = "#89b4fa"
DIM      = "#6c7086"
CROWN    = "#f9e2af"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CHOICE_RE = re.compile(r"(\d+)\s*[.\-:)]?\s*([a-zA-Z]+)?")


def parse_choices(text: str):
    """Parse '1. Go 3. Go' or '1.Y 2.N' into [(idx, word), ...] (0-indexed)."""
    out = []
    for num, word in _CHOICE_RE.findall(text):
        out.append((int(num) - 1, (word or "").lower()))
    return out


def parse_yn(w: str) -> bool:
    w = w.lower()
    if w in ("y", "yes", "1", "true", "t", "approve", "ok"):
        return True
    if w in ("n", "no", "0", "false", "f", "reject"):
        return False
    raise ValueError(f"can't parse vote {w!r} (use Y/N)")


def parse_sf(w: str) -> bool:
    w = w.lower()
    if w in ("s", "succeed", "success", "y", "yes", "1", "t", "true", "good"):
        return True
    if w in ("f", "fail", "failure", "n", "no", "0", "false", "bad"):
        return False
    raise ValueError(f"can't parse play {w!r} (use S/F)")


def open_role(game: GameState, viewer: int, target: int) -> str:
    """The role label shown for `target` from `viewer`'s perspective."""
    if target == viewer:
        return f"{game.assignment[viewer].name} (you)"
    me = game.assignment[viewer]
    them = game.assignment[target]
    if me is Role.MERLIN or me.is_evil():
        return "EVIL" if them.is_evil() else "?"
    return "?"


# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

class Tooltip:
    def __init__(self, parent):
        self.parent = parent
        self._win = None

    def show(self, x_root: int, y_root: int, text: str):
        self.hide()
        tw = tk.Toplevel(self.parent)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x_root + 14}+{y_root + 14}")
        tk.Label(
            tw, text=text, bg="#313244", fg=FG,
            relief="solid", borderwidth=1, padx=6, pady=3,
            font=("Consolas", 10),
        ).pack()
        self._win = tw

    def hide(self):
        if self._win is not None:
            self._win.destroy()
            self._win = None


# ---------------------------------------------------------------------------
# Debug window: hidden info, consistency mask, belief posterior
# ---------------------------------------------------------------------------

class DebugWindow:
    """Toplevel window with three tabs that auto-refresh on game state changes.

    Tabs:
      - Hidden Info : true assignment + per-round private/public history.
      - Consistency : the (60,) bool mask from src.consistency.
      - Beliefs     : per-player marginals + top-K assignments by posterior.
    Closing the window only hides it; reopen via the Debug button.
    """

    def __init__(self, parent_gui: "AvalonGUI"):
        self.parent = parent_gui
        self.win = tk.Toplevel(parent_gui.root)
        self.win.title("Avalon Debug")
        self.win.configure(bg=BG)
        self.win.geometry("780x640")
        self.win.protocol("WM_DELETE_WINDOW", self.hide)

        style = ttk.Style(self.win)
        try:
            style.theme_use("default")
        except tk.TclError:
            pass
        style.configure(
            "Debug.TNotebook", background=BG, borderwidth=0,
            tabmargins=[2, 6, 2, 0],
        )
        style.configure(
            "Debug.TNotebook.Tab", background=PANEL, foreground=DIM,
            padding=[14, 6], font=("Consolas", 10), borderwidth=0,
        )
        style.map(
            "Debug.TNotebook.Tab",
            background=[("selected", "#313244")],
            foreground=[("selected", ACCENT)],
        )

        nb = ttk.Notebook(self.win, style="Debug.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.info_text = self._make_tab(nb, "Hidden Info")
        self.cons_text = self._make_tab(nb, "Consistency")
        self.belief_text = self._make_tab(nb, "Beliefs")

        self.win.withdraw()

    def _make_tab(self, nb: ttk.Notebook, title: str) -> tk.Text:
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text=title)
        text = tk.Text(
            frame, bg=PANEL, fg=FG, font=("Consolas", 10),
            wrap="none", relief="flat", insertbackground=FG,
            padx=8, pady=6,
        )
        sb = tk.Scrollbar(frame, command=text.yview, bg=BG)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.tag_configure("truth", foreground=ACCENT)
        text.tag_configure("good", foreground=GOOD)
        text.tag_configure("evil", foreground=EVIL)
        text.tag_configure("dim", foreground=DIM)
        text.configure(state="disabled")
        return text

    # ---- visibility ----
    def toggle(self):
        if self._is_visible():
            self.hide()
        else:
            self.show()

    def show(self):
        self.win.deiconify()
        self.refresh()

    def hide(self):
        self.win.withdraw()

    def _is_visible(self) -> bool:
        try:
            return self.win.state() == "normal"
        except tk.TclError:
            return False

    # ---- refresh ----
    def refresh(self):
        if not self._is_visible():
            return
        game = self.parent.game
        if game is None:
            return
        obs = game.observation()
        self._render_hidden(game)
        try:
            mask = consistency_mask(obs)
        except Exception as e:
            self._set(self.cons_text, f"consistency_mask error: {e}")
            mask = None
        if mask is not None:
            self._render_consistency(mask, game)
        try:
            tracker = BeliefTracker()
            tracker.observe(obs)
            self._render_beliefs(tracker, game)
        except Exception as e:
            self._set(self.belief_text, f"belief computation error: {e}")

    @staticmethod
    def _set(widget: tk.Text, content):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        if isinstance(content, list):
            for chunk, tag in content:
                widget.insert(tk.END, chunk, tag) if tag else widget.insert(tk.END, chunk)
        else:
            widget.insert(tk.END, content)
        widget.configure(state="disabled")

    # ---- rendering ----
    def _render_hidden(self, game: GameState):
        chunks = []
        chunks.append(("=== Hidden Game Info ===\n\n", "truth"))
        chunks.append(("Assignment:\n", None))
        for i, role in enumerate(game.assignment):
            tag = "evil" if role.is_evil() else "good"
            you = " (you)" if i == USER_PLAYER else ""
            chunks.append((f"  P{i+1}: {role.name}{you}\n", tag))
        evils = [i + 1 for i, r in enumerate(game.assignment) if r.is_evil()]
        merlin = next(i + 1 for i, r in enumerate(game.assignment) if r is Role.MERLIN)
        assassin = next(i + 1 for i, r in enumerate(game.assignment) if r is Role.ASSASSIN)
        chunks.append((f"\nEvil seats: {evils}\n", "evil"))
        chunks.append((f"Merlin:     P{merlin}\n", "good"))
        chunks.append((f"Assassin:   P{assassin}\n", "evil"))
        chunks.append((f"\nPhase: {game.phase.value.upper()}    "
                       f"Score: {game.successes}S / {game.failures}F    "
                       f"Rejections: {game.rejected_proposals}/{MAX_PROPOSALS}\n",
                       "dim"))

        for r_idx, rec in enumerate(game.rounds):
            chunks.append((
                f"\n--- Round {r_idx+1} (team size {TEAM_SIZES[r_idx]}) ---\n",
                "dim",
            ))
            for p_i, prop in enumerate(rec.proposals):
                team_str = ",".join(f"P{p+1}" for p in prop.team)
                if prop.votes is not None:
                    vote_str = " ".join("Y" if v else "N" for v in prop.votes)
                    yc = sum(prop.votes)
                    nc = NUM_PLAYERS - yc
                    outcome = "APPROVED" if prop.approved else "REJECTED"
                    chunks.append((
                        f"  Proposal {p_i+1}: by P{prop.proposer+1}, "
                        f"team [{team_str}]\n"
                        f"    votes: {vote_str}  ({yc}-{nc} {outcome})\n",
                        None,
                    ))
                else:
                    chunks.append((
                        f"  Proposal {p_i+1}: by P{prop.proposer+1}, "
                        f"team [{team_str}] (awaiting vote)\n",
                        None,
                    ))
            if rec.mission_fails is not None:
                tag = "good" if rec.succeeded else "evil"
                verdict = "SUCCESS" if rec.succeeded else f"FAIL ({rec.mission_fails} fails)"
                chunks.append((f"  Mission: {verdict}\n", tag))

        if game.is_terminal():
            tag = "good" if game.winner == "arthur" else "evil"
            chunks.append((f"\nWinner: {game.winner.upper()}\n", tag))
            if game.assassin_target is not None:
                chunks.append((f"Assassin shot: P{game.assassin_target+1}\n", "evil"))
        self._set(self.info_text, chunks)

    def _render_consistency(self, mask: np.ndarray, game: GameState):
        truth = tuple(game.assignment)
        truth_idx = ASSIGNMENTS.index(truth)
        consistent = int(mask.sum())
        chunks = [
            (f"Consistent assignments: {consistent} / {len(mask)}\n", "truth"),
            ("(★ = true assignment)\n\n", "dim"),
            (f"  {'idx':>4}  {'s1':<3}{'s2':<3}{'s3':<3}{'s4':<3}{'s5':<3}  "
             f"{'evil':<10}  status\n", "dim"),
            ("  " + "-" * 50 + "\n", "dim"),
        ]
        for a_idx, (rho, valid) in enumerate(zip(ASSIGNMENTS, mask)):
            mark = "★ " if a_idx == truth_idx else "  "
            cells = "".join(f"{r.name[0]:<3}" for r in rho)
            evils = [i + 1 for i in sorted(evil_indices(rho))]
            status = "OK" if valid else "x"
            tag = "truth" if a_idx == truth_idx else (None if valid else "dim")
            chunks.append((
                f"{mark}{a_idx:>4}  {cells}  {str(evils):<10}  {status}\n",
                tag,
            ))
        self._set(self.cons_text, chunks)

    def _render_beliefs(self, tracker: BeliefTracker, game: GameState):
        belief = tracker.belief
        truth = tuple(game.assignment)
        truth_idx = ASSIGNMENTS.index(truth)

        chunks = [("=== Per-player marginals ===\n\n", "truth")]
        chunks.append((
            f"        {'LS':>7}{'Merlin':>9}{'DS':>7}"
            f"{'Assassin':>11}{'Evil':>9}\n",
            "dim",
        ))
        for p in range(NUM_PLAYERS):
            ls = tracker.marginal_role(p, Role.LS)
            m = tracker.marginal_role(p, Role.MERLIN)
            ds = tracker.marginal_role(p, Role.DS)
            a = tracker.marginal_role(p, Role.ASSASSIN)
            e = tracker.marginal_evil(p)
            tag = "truth" if p == USER_PLAYER else None
            chunks.append((
                f"  P{p+1}    {ls:>7.3f}{m:>9.3f}{ds:>7.3f}{a:>11.3f}{e:>9.3f}\n",
                tag,
            ))

        chunks.append(("\n=== Top assignments by posterior ===\n", "truth"))
        chunks.append(("(★ = true assignment)\n\n", "dim"))
        order = np.argsort(-belief)
        shown = 0
        for a_idx in order:
            p = float(belief[a_idx])
            if p < 1e-9 and shown >= 5:
                break
            roles = " ".join(r.name[0] for r in ASSIGNMENTS[a_idx])
            evils = [i + 1 for i in sorted(evil_indices(ASSIGNMENTS[a_idx]))]
            mark = "★ " if a_idx == truth_idx else "  "
            tag = "truth" if a_idx == truth_idx else None
            chunks.append((
                f"{mark}#{shown+1:>2}  p={p:.4f}  ρ=[{roles}]   evil={evils}\n",
                tag,
            ))
            shown += 1
            if shown >= 20:
                break
        chunks.append((
            f"\nTrue ρ posterior: p={float(belief[truth_idx]):.4f}\n",
            "truth",
        ))
        self._set(self.belief_text, chunks)


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class AvalonGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Avalon — Solo Test GUI")
        self.root.configure(bg=BG)
        self.rng = random.Random()
        self.game: GameState = None  # set by _new_game
        self._build_ui()
        self.debug = DebugWindow(self)
        self.debug_btn.configure(command=self.debug.toggle)
        self._new_game()

    # ----- construction -----
    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        self.status_var = tk.StringVar()
        tk.Label(
            top, textvariable=self.status_var, bg=BG, fg=FG,
            font=("Consolas", 11), anchor="w", justify="left",
        ).pack(fill=tk.X)

        self.canvas = tk.Canvas(
            self.root, width=660, height=600,
            bg=BG, highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=4)
        self.tooltip = Tooltip(self.canvas)

        log_frame = tk.Frame(self.root, bg=BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.log = tk.Text(
            log_frame, height=9, bg=PANEL, fg=FG,
            font=("Consolas", 10), wrap="word",
            insertbackground=FG, relief="flat",
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state="disabled")

        entry_frame = tk.Frame(self.root, bg=BG)
        entry_frame.pack(fill=tk.X, padx=10, pady=(4, 10))
        tk.Label(entry_frame, text=">", bg=BG, fg=ACCENT,
                 font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self.entry = tk.Entry(
            entry_frame, bg=PANEL, fg=FG,
            font=("Consolas", 11),
            insertbackground=FG, relief="flat",
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, ipady=4)
        self.entry.bind("<Return>", self._on_submit)

        tk.Button(
            entry_frame, text="New Game", command=self._new_game,
            bg="#313244", fg=FG, relief="flat",
            font=("Consolas", 10), padx=10, pady=2,
            activebackground="#45475a", activeforeground=FG,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        self.debug_btn = tk.Button(
            entry_frame, text="Debug", command=lambda: None,
            bg="#313244", fg=FG, relief="flat",
            font=("Consolas", 10), padx=10, pady=2,
            activebackground="#45475a", activeforeground=FG,
        )
        self.debug_btn.pack(side=tk.RIGHT, padx=(6, 0))

    # ----- game control -----
    def _new_game(self):
        self.game = new_game(rng=self.rng)
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")
        self._log(
            f"New game. You are P{USER_PLAYER+1}: "
            f"{self.game.assignment[USER_PLAYER].name}.",
            ACCENT,
        )
        self._log(
            "Hidden assignment (debug): "
            + str([r.name for r in self.game.assignment]),
            DIM,
        )
        self._draw()
        self._prompt()
        self.entry.focus_set()

    # ----- logging -----
    def _log(self, text: str, color: str = FG):
        tag = "c_" + color.replace("#", "")
        self.log.configure(state="normal")
        self.log.tag_configure(tag, foreground=color)
        self.log.insert(tk.END, text + "\n", tag)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    # ----- drawing -----
    def _draw(self):
        self.status_var.set(self._status_text())
        c = self.canvas
        c.delete("all")
        self._draw_round_indicators()
        cx, cy, r = 330, 320, 175
        for i in range(NUM_PLAYERS):
            angle = -math.pi / 2 + i * 2 * math.pi / NUM_PLAYERS
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            self._draw_humanoid(x, y, i)
        if hasattr(self, "debug"):
            self.debug.refresh()

    def _draw_round_indicators(self):
        c = self.canvas
        cx_center, y, radius = 330, 38, 18
        spacing = 60
        start_x = cx_center - spacing * (NUM_ROUNDS - 1) / 2
        for r_idx in range(NUM_ROUNDS):
            cx = start_x + r_idx * spacing
            rec = self.game.rounds[r_idx] if r_idx < len(self.game.rounds) else None
            if rec is not None and rec.succeeded is True:
                outline, fill = GOOD, GOOD
            elif rec is not None and rec.succeeded is False:
                outline, fill = EVIL, EVIL
            else:
                outline, fill = DIM, ""
            c.create_oval(cx - radius, y - radius, cx + radius, y + radius,
                          outline=outline, fill=fill, width=2)
            c.create_text(cx, y, text=str(TEAM_SIZES[r_idx]),
                          fill=FG if not fill else PANEL,
                          font=("Consolas", 10, "bold"))
            c.create_text(cx, y + radius + 12, text=f"R{r_idx+1}",
                          fill=DIM, font=("Consolas", 9))

    def _draw_humanoid(self, x: float, y: float, idx: int):
        c = self.canvas
        g = self.game

        # team membership for currently-active proposal (during VOTE or MISSION)
        on_team = False
        if g.phase in (Phase.VOTE, Phase.MISSION) and g.current_proposal is not None:
            on_team = idx in g.current_proposal.team

        role_label = open_role(g, USER_PLAYER, idx)
        if idx == USER_PLAYER:
            color = ACCENT
        elif "EVIL" in role_label:
            color = EVIL
        else:
            color = NEUTRAL

        outline = GOOD if on_team else color
        width = 3 if on_team else 2

        head = c.create_oval(x - 16, y - 30, x + 16, y + 2,
                             outline=outline, width=width, fill=PANEL)
        body = c.create_line(x, y + 2, x, y + 44, fill=outline, width=width)
        arms = c.create_line(x - 22, y + 18, x + 22, y + 18, fill=outline, width=width)
        leg1 = c.create_line(x, y + 44, x - 16, y + 72, fill=outline, width=width)
        leg2 = c.create_line(x, y + 44, x + 16, y + 72, fill=outline, width=width)

        # crown over current proposer (only meaningful in PROPOSE phase)
        if idx == g.proposer and g.phase is Phase.PROPOSE:
            c.create_text(x, y - 46, text="♛", fill=CROWN,
                          font=("Arial", 18, "bold"))

        # assassin marker during ASSASSINATE
        if g.phase is Phase.ASSASSINATE and g.assignment[idx] is Role.ASSASSIN:
            c.create_text(x, y - 46, text="☠", fill=EVIL,
                          font=("Arial", 16, "bold"))

        c.create_text(x, y + 92, text=f"P{idx+1}", fill=FG,
                      font=("Consolas", 11, "bold"))
        c.create_text(x, y + 108, text=role_label, fill=color,
                      font=("Consolas", 10))

        hidden = g.assignment[idx].name
        tip = f"P{idx+1} hidden role: {hidden}"
        for it in (head, body, arms, leg1, leg2):
            c.tag_bind(it, "<Enter>",
                       lambda e, t=tip: self.tooltip.show(e.x_root, e.y_root, t))
            c.tag_bind(it, "<Leave>", lambda e: self.tooltip.hide())

    def _status_text(self) -> str:
        g = self.game
        lines = [
            f"Round {min(g.round_idx + 1, 5)}/5   "
            f"Successes: {g.successes}   Failures: {g.failures}   "
            f"Rejections: {g.rejected_proposals}/{MAX_PROPOSALS}",
            f"Phase: {g.phase.value.upper()}   "
            f"Proposer: P{g.proposer + 1}   "
            f"Team size this round: {TEAM_SIZES[min(g.round_idx, NUM_PLAYERS-1)]}",
        ]
        if g.is_terminal():
            lines.append(f"WINNER: {g.winner.upper()}")
        return "\n".join(lines)

    # ----- prompts -----
    def _prompt(self):
        g = self.game
        if g.is_terminal():
            self._log(f"Game over. Winner: {g.winner.upper()}", ACCENT)
            return
        if g.phase is Phase.PROPOSE:
            k = TEAM_SIZES[g.round_idx]
            self._log(
                f"[PROPOSE] P{g.proposer+1} picks a team of {k}. "
                f"e.g. '1. Go 3. Go'",
                ACCENT,
            )
        elif g.phase is Phase.VOTE:
            team = ", ".join(f"P{p+1}" for p in g.current_proposal.team)
            self._log(
                f"[VOTE] P{g.proposer+1} put up team [{team}]. "
                f"List YES voters, e.g. '1 3 5' (P2,P4 vote NO). "
                f"'+' = all YES, '-' = all NO.",
                ACCENT,
            )
        elif g.phase is Phase.MISSION:
            team = ", ".join(f"P{p+1}" for p in g.current_proposal.team)
            self._log(
                f"[MISSION] Team [{team}] plays. "
                f"List who SUCCEEDS (rest fail). Good must succeed. "
                f"'+' = all SUCCEED, '-' = all FAIL.",
                ACCENT,
            )
        elif g.phase is Phase.ASSASSINATE:
            assassin = next(i for i, r in enumerate(g.assignment) if r is Role.ASSASSIN)
            self._log(
                f"[ASSASSINATE] Assassin is P{assassin+1}. "
                f"Pick a target (e.g. '3'). Hit Merlin -> Mordred wins.",
                ACCENT,
            )

    # ----- input -----
    def _on_submit(self, _evt=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._log(f"> {text}", DIM)
        try:
            self._handle_input(text)
        except (AssertionError, ValueError) as e:
            self._log(f"!! {e}", EVIL)
            return
        self._draw()
        self._prompt()

    def _handle_input(self, text: str):
        g = self.game
        if g.is_terminal():
            return

        if g.phase is Phase.PROPOSE:
            choices = parse_choices(text)
            team = tuple(sorted({idx for idx, _ in choices}))
            k = TEAM_SIZES[g.round_idx]
            if len(team) != k:
                raise ValueError(f"need exactly {k} distinct players, got {len(team)}")
            if not all(0 <= p < NUM_PLAYERS for p in team):
                raise ValueError("player numbers must be 1..5")
            g.propose(team)
            self._log(f"Team proposed: {[p+1 for p in team]}")

        elif g.phase is Phase.VOTE:
            stripped = text.strip().lower()
            if stripped in ("+", "all", "yes", "y"):
                votes = [True] * NUM_PLAYERS
            elif stripped in ("-", "none", "no", "n"):
                votes = [False] * NUM_PLAYERS
            else:
                choices = parse_choices(text)
                has_words = any(w for _, w in choices)
                if has_words:
                    votes = [None] * NUM_PLAYERS
                    for idx, w in choices:
                        if 0 <= idx < NUM_PLAYERS and w:
                            votes[idx] = parse_yn(w)
                    missing = [i + 1 for i, v in enumerate(votes) if v is None]
                    if missing:
                        raise ValueError(f"need yes/no for players {missing}")
                else:
                    yes_set = {idx for idx, _ in choices if 0 <= idx < NUM_PLAYERS}
                    votes = [i in yes_set for i in range(NUM_PLAYERS)]
            g.vote(tuple(votes))
            yc = sum(votes)
            nc = NUM_PLAYERS - yc
            outcome = "APPROVED" if yc > nc else "REJECTED"
            self._log(
                f"Votes: {['Y' if v else 'N' for v in votes]}  "
                f"-> {yc}-{nc} {outcome}"
            )

        elif g.phase is Phase.MISSION:
            team = g.current_proposal.team
            stripped = text.strip().lower()
            if stripped in ("+", "all", "succeed", "success", "s"): # for debugging purposes: all or none is accepted 
                plays = {p: True for p in team}
            elif stripped in ("-", "none", "fail", "f"): 
                plays = {p: False for p in team}
            else:
                choices = parse_choices(text)
                has_words = any(w for _, w in choices)
                if has_words:
                    plays = {}
                    for idx, w in choices:
                        if idx in team and w:
                            plays[idx] = parse_sf(w)
                    missing = [p + 1 for p in team if p not in plays]
                    if missing:
                        raise ValueError(f"need play (S/F) for team members {missing}")
                else:
                    succeed_set = {idx for idx, _ in choices if idx in team}
                    plays = {p: (p in succeed_set) for p in team}
            prev_round_idx = g.round_idx
            g.play_mission(plays)
            rec = g.rounds[prev_round_idx]
            verdict = "SUCCESS" if rec.succeeded else f"FAIL ({rec.mission_fails} fails)"
            self._log(f"Mission round {prev_round_idx+1}: {verdict}")

        elif g.phase is Phase.ASSASSINATE:
            choices = parse_choices(text)
            if not choices:
                raise ValueError("need a target player number")
            target = choices[0][0]
            if not 0 <= target < NUM_PLAYERS:
                raise ValueError("target must be 1..5")
            assassin = next(i for i, r in enumerate(g.assignment) if r is Role.ASSASSIN)
            if target == assassin:
                raise ValueError("assassin can't target themselves")
            hit_merlin = g.assignment[target] is Role.MERLIN
            g.assassinate(target)
            self._log(
                f"Assassin shot P{target+1} "
                f"({'MERLIN — Mordred wins' if hit_merlin else 'not Merlin — Arthur wins'})"
            )


def main():
    root = tk.Tk()
    AvalonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
