"""Step 2 SPEC §6: policy-agnostic episode scoring."""

import dataclasses

import pytest

from toycar import Info, score_episode, summarize

BASE = Info(station_m=0.0, progress_m=0.0, laps=0, direction="cw", wrong_way=False, branch=None,
            last_branch=None, passages=0, steps=0, termination=None)


def stream(start_branch, events, n=40, termination="max_steps", wrong_way_at=()):
    """Synthetic Info stream: `events` maps step k → branch of a passage completed at step k."""
    infos = [dataclasses.replace(BASE, branch=start_branch)]
    passages, last = 0, None
    for k in range(1, n + 1):
        if k in events:
            passages, last = passages + 1, events[k]
        infos.append(dataclasses.replace(
            BASE, steps=k, passages=passages, last_branch=last, wrong_way=k in wrong_way_at,
            termination=termination if k == n else None, laps=k // 20, progress_m=float(k),
        ))
    return infos


def test_first_passage_exempt_only_inside_section(track):
    events = {5: "inner", 20: "outer", 35: "outer"}
    inside = score_episode(track, "outer", stream("inner", events))
    assert inside.passages == ("inner", "outer", "outer")
    assert (inside.scored, inside.correct) == (2, 2)
    assert inside.instruction_followed and inside.success

    outside = score_episode(track, "outer", stream(None, events))
    assert (outside.scored, outside.correct) == (3, 2)
    assert not outside.instruction_followed and not outside.success


def test_wrong_branch_under_outer(track):
    sc = score_episode(track, "outer", stream(None, {10: "outer", 30: "inner"}))
    assert sc.scored == 2 and sc.correct == 1
    assert not sc.instruction_followed


def test_wrong_way_ticks(track):
    sc = score_episode(track, "inner", stream(None, {10: "inner"}, wrong_way_at=(3, 4, 5)))
    assert sc.wrong_way_ticks == 3
    assert not sc.instruction_followed


def test_any_counts_every_passage(track):
    sc = score_episode(track, "any", stream(None, {5: "outer", 15: "inner", 25: "inner"}))
    assert sc.scored == sc.correct == 3
    assert sc.instruction_followed


def test_crash_and_final_fields(track):
    sc = score_episode(track, "outer", stream(None, {10: "outer"}, n=30, termination="crash"))
    assert sc.crashed and not sc.success and sc.instruction_followed
    assert (sc.steps, sc.laps, sc.progress_m, sc.termination, sc.direction) == (30, 1, 30.0, "crash", "cw")


def test_empty_raises(track):
    with pytest.raises(ValueError):
        score_episode(track, "any", [])


def test_summarize(track):
    scores = [
        score_episode(track, "any", stream("outer", {5: "outer", 15: "inner", 25: "outer"})),
        score_episode(track, "outer", stream(None, {5: "outer", 15: "inner"}, termination="crash")),
    ]
    sm = summarize(scores)
    assert sm["episodes"] == 2
    assert sm["crash_rate"] == 0.5
    assert sm["scored_passages"] == 4
    assert sm["wrong_passages"] == 1
    assert sm["outer_fraction"] == 0.5  # any episode: first exempt → (inner, outer)
    assert sm["success_rate"] == 0.5
    assert summarize(scores[1:])["outer_fraction"] is None
