# capsa_service_network.py
# CapSA-based hub network design with DC allocation & hub replacement/merging
# Outputs: capsa_service_output.xlsx with multiple sheets (now includes iterations sheet)

import os, json, random
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
CFG = {
    "excel_path": r"C:\Users\VENKAT\Desktop\old output.xlsx",  # Input file with Pins, DCs, Hubs
    "sheet_pins": "Pins",
    "sheet_dcs": "DCs",
    "sheet_hubs": "Hubs",

    # Service thresholds
    "R30": 30.0,
    "R50": 50.0,
    "D_MAX": 150.0,

    # Viability
    "VIABLE_KG": 20000.0,

    # Candidate selection
    "pin_candidate_mode": "top_n",
    "pin_candidates_top_n": 800,
    "grid_cell_deg": 0.25,

    # CapSA params
    "pop_size": 60,
    "iters": 500,
    "phi_elite": 0.6,
    "phi_peer": 0.3,
    "noise": 0.2,
    "flip_rate": 0.03,
    "anneal": 0.995,
    "seed": 42,

    "out_dir": "out",

    # Objective weights
    "W30": 1_000_000.0,
    "W50": 10_000.0,
    "LAMBDA_HUBS": 50_000.0,
    "LAMBDA_SEP": 100_000.0,
    "R_SEP": 25.0,

    # Replacement/Merge radius
    "REPLACE_RADIUS": 5.0
}

# ---------------- HELPERS ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True) if not os.path.exists(p) else None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def load_inputs(cfg):
    xls = pd.ExcelFile(cfg["excel_path"])
    pins = pd.read_excel(xls, cfg["sheet_pins"])
    dcs  = pd.read_excel(xls, cfg["sheet_dcs"])
    hubs = pd.read_excel(xls, cfg["sheet_hubs"])  # Old hubs
    pins.columns = [c.lower().strip() for c in pins.columns]
    dcs.columns = [c.lower().strip() for c in dcs.columns]
    hubs.columns = [c.lower().strip() for c in hubs.columns]

    # weight
    if "avg_weight_kg" in pins.columns:
        pins["weight_kg"] = pins["avg_weight_kg"].astype(float)
    else:
        mcols = [c for c in pins.columns if c.startswith("wt_") or c.startswith("weight_")]
        pins["weight_kg"] = pins[mcols].astype(float).replace(0, np.nan).mean(axis=1).fillna(0)

    # ensure required pin fields exist
    for f in ("pincode","lat","lon","weight_kg"):
        if f not in pins.columns:
            raise ValueError(f"Pins sheet must contain column: {f}")

    # hubs (existing)
    if "hub_id" not in hubs.columns:
        hubs["hub_id"] = hubs.index.astype(str)
    hubs["hub_type"] = "present"; hubs["is_dc"] = False
    hubs = hubs[["hub_id","name","lat","lon","hub_type","is_dc"]]

    # DCs as hubs (always open)
    dcs_h = dcs.copy()
    # expected columns in DCs: dc_id, dc_name, lat, lon
    if not set(["dc_id","dc_name","lat","lon"]).issubset(set(dcs_h.columns)):
        raise ValueError("DCs sheet must contain columns: dc_id, dc_name, lat, lon")
    dcs_h["hub_id"] = dcs_h["dc_id"].astype(str)
    dcs_h["name"] = dcs_h["dc_name"]
    dcs_h["hub_type"] = "dc"; dcs_h["is_dc"] = True
    dcs_h = dcs_h[["hub_id","name","lat","lon","hub_type","is_dc"]]

    hubs_all = pd.concat([hubs, dcs_h], ignore_index=True)
    return pins, dcs, hubs, hubs_all

def add_pin_candidates(pins, hubs_all, cfg):
    if cfg["pin_candidate_mode"] == "all":
        cand = pins.copy()
    else:
        cand = pins.sort_values("weight_kg", ascending=False).head(cfg["pin_candidates_top_n"]).copy()

    cell = cfg["grid_cell_deg"]
    cand["_cell_lat"] = (cand["lat"] / cell).round().astype(int)
    cand["_cell_lon"] = (cand["lon"] / cell).round().astype(int)
    cand = cand.sort_values("weight_kg", ascending=False).drop_duplicates(subset=["_cell_lat","_cell_lon"])

    if "pincode" not in cand.columns:
        raise ValueError("Pins sheet must include 'pincode' column to create PIN candidates.")

    cand["hub_id"] = "PIN_" + cand["pincode"].astype(str)
    cand["name"] = cand["hub_id"]; cand["hub_type"] = "pin_candidate"; cand["is_dc"] = False
    cand = cand[["hub_id","name","lat","lon","hub_type","is_dc"]]

    hubs_all = pd.concat([hubs_all, cand], ignore_index=True).drop_duplicates(subset=["hub_id"])
    return hubs_all.reset_index(drop=True)

def distances(hubs,pins):
    H = hubs.shape[0]; P = pins.shape[0]
    dist = np.empty((H,P), dtype=float)
    for h in range(H):
        dist[h,:] = haversine_km(hubs.lat.iloc[h], hubs.lon.iloc[h], pins.lat.to_numpy(), pins.lon.to_numpy())
    return dist

def hubhub_dist(hubs):
    H = hubs.shape[0]
    out = np.empty((H,H), dtype=float)
    for i in range(H):
        out[i,:] = haversine_km(hubs.lat.iloc[i], hubs.lon.iloc[i], hubs.lat.to_numpy(), hubs.lon.to_numpy())
    return out

# ---------------- ASSIGNMENT ----------------
def assign_and_score(open_mask, dist_hp, pins, hubs, cfg):
    weight = pins["weight_kg"].to_numpy()
    H,P = dist_hp.shape
    open_idx = np.where(open_mask)[0]
    chosen_h = np.full(P, -1, dtype=int)
    chosen_d = np.full(P, np.inf, dtype=float)

    if open_idx.size > 0:
        for start in range(0, P, 20000):
            end = min(P, start+20000)
            dblock = dist_hp[open_idx, start:end]     # shape (n_open, chunk)
            best = np.argmin(dblock, axis=0)          # indices into open_idx
            d = dblock[best, np.arange(dblock.shape[1])]
            h = open_idx[best]
            mask = d <= cfg["D_MAX"]
            chosen_h[start:end][mask] = h[mask]
            chosen_d[start:end][mask] = d[mask]

    vol = np.zeros(H, dtype=float)
    np.add.at(vol, chosen_h[chosen_h >= 0], weight[chosen_h >= 0])

    # enforce hard viability: close hubs <VIABLE_KG and reassign
    close = np.where((~hubs.is_dc) & (vol < cfg["VIABLE_KG"]) & open_mask)[0]
    if close.size:
        open_mask[close] = False
        return assign_and_score(open_mask, dist_hp, pins, hubs, cfg)

    # service KPIs
    total_w = np.sum(weight) if np.sum(weight)>0 else 1.0
    pct30 = 100.0 * np.sum(weight[chosen_d <= cfg["R30"]]) / total_w
    pct50 = 100.0 * np.sum(weight[chosen_d <= cfg["R50"]]) / total_w

    return {
        "pct30": float(pct30),
        "pct50": float(pct50),
        "open_mask": open_mask.copy(),
        "chosen_h": chosen_h,
        "chosen_d": chosen_d,
        "vol": vol
    }

# ---------------- OBJECTIVE ----------------
def objective(res, hubs, cfg, dist_hh):
    base = -(cfg["W30"] * res["pct30"] + cfg["W50"] * res["pct50"])
    open_mask = res["open_mask"]; is_dc = hubs.is_dc.to_numpy()
    n_open = int(np.sum(open_mask & (~is_dc)))
    hub_pen = cfg["LAMBDA_HUBS"] * n_open

    idx = np.where(open_mask & (~is_dc))[0]; sep_pairs = 0
    if idx.size > 1:
        dh = dist_hh[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        sep_pairs = int(np.sum(dh[iu] < cfg["R_SEP"]))
    sep_pen = cfg["LAMBDA_SEP"] * sep_pairs

    return base + hub_pen + sep_pen

# ---------------- CapSA ----------------
def capsa(dist_hp, dist_hh, pins, hubs, cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    H = hubs.shape[0]; is_dc = hubs.is_dc.to_numpy(); dc_idx = np.where(is_dc)[0]
    def sigmoid(x): return 1/(1+np.exp(-x))

    pop = []
    for _ in range(cfg["pop_size"]):
        g = np.random.normal(0,1,H)
        non_dc = np.where(~is_dc)[0]
        if non_dc.size:
            k = max(1, int(0.15 * non_dc.size))
            g[np.random.choice(non_dc, k, replace=False)] += 2.0
        pop.append(g)

    elite_score = np.inf; elite_g = None; elite_res = None
    noise = cfg["noise"]

    # iteration records
    iter_records = []
    best_obj_so_far = np.inf

    for it in range(cfg["iters"]):
        scored = []
        for g in pop:
            g_eval = g.copy(); g_eval[dc_idx] = 6.0   # force DCs open
            open_mask = sigmoid(g_eval) > 0.5
            res = assign_and_score(open_mask, dist_hp, pins, hubs, cfg)
            score = objective(res, hubs, cfg, dist_hh)
            scored.append((float(score), g_eval, res))

        # sort ascending (better = lower)
        scored.sort(key=lambda x: x[0])

        # best of this iteration
        top_score, top_g, top_res = scored[0]
        if top_score < best_obj_so_far:
            best_obj_so_far = top_score

        # update global elite if improved
        if scored[0][0] < elite_score:
            elite_score, elite_g, elite_res = scored[0]

        # record iteration metrics (best candidate this iteration)
        pct30_it = float(top_res.get("pct30", np.nan))
        pct50_it = float(top_res.get("pct50", np.nan))
        hubs_open_it = int(np.sum(top_res["open_mask"]))
        iter_records.append({
            "iteration": it+1,
            "obj": float(top_score),
            "best_obj_so_far": float(best_obj_so_far),
            "pct30": pct30_it,
            "pct50": pct50_it,
            "hubs_open": hubs_open_it
        })

        # create new population (simple update using elite + peer mixing + noise)
        new_pop = []
        for _, g, res in scored:
            # peer: pick random scored member's evaluated vector (the g_eval from tuple)
            peer = random.choice(scored)[1]
            g_new = g + cfg["phi_elite"] * (elite_g - g) + cfg["phi_peer"] * (peer - g) + noise * np.random.normal(0,1,g.shape)
            flips = np.random.rand(H) < cfg["flip_rate"]
            if flips.any():
                g_new[flips] += np.random.choice([-3, +3], np.sum(flips))
            g_new[dc_idx] = 6.0
            new_pop.append(g_new)
        pop = new_pop
        noise *= cfg["anneal"]

        # periodic console logging (safe guard if elite_res not set yet)
        if (it+1) % 20 == 0:
            display_res = elite_res if elite_res is not None else top_res
            disp_pct30 = float(display_res.get("pct30", np.nan))
            disp_pct50 = float(display_res.get("pct50", np.nan))
            print(f"[iter {it+1}] obj={best_obj_so_far:.0f}  {disp_pct30:.1f}%≤30  {disp_pct50:.1f}%≤50  hubs={int(np.sum(display_res['open_mask']))}")

    # If algorithm never found elite_res (shouldn't happen), take best from last iteration
    if elite_res is None and iter_records:
        # reconstruct from last iter record using last scored top (not ideal, but safe)
        last = iter_records[-1]
        # can't reconstruct chosen_h/chosen_d/vol here; raise if needed
        raise RuntimeError("CapSA failed to produce an elite solution. Check inputs/parameters.")

    return elite_res, iter_records, float(best_obj_so_far)

# ---------------- HUB MERGING ----------------
def merge_hubs(hubs_sel, cfg):
    merged = []
    replacements = []

    hubs_sel = hubs_sel.sort_values("assigned_kg", ascending=False).reset_index(drop=True)
    used = np.zeros(len(hubs_sel), dtype=bool)

    for i, row in hubs_sel.iterrows():
        if used[i]:
            continue
        base_hub = row.to_dict()
        used[i] = True

        for j in range(i+1, len(hubs_sel)):
            if used[j]:
                continue
            dist = haversine_km(row["lat"], row["lon"], hubs_sel.loc[j, "lat"], hubs_sel.loc[j, "lon"])
            if dist < cfg["REPLACE_RADIUS"]:
                replacements.append({
                    "kept_hub_id": row["hub_id"],
                    "kept_hub_name": row["name"],
                    "merged_hub_id": hubs_sel.loc[j, "hub_id"],
                    "merged_hub_name": hubs_sel.loc[j, "name"],
                    "distance_km": float(dist)
                })
                used[j] = True
        merged.append(base_hub)

    return pd.DataFrame(merged), pd.DataFrame(replacements)

# ---------------- MAIN ----------------
def main():
    cfg = CFG.copy(); ensure_dir(cfg["out_dir"])
    pins, dcs, old_hubs, hubs = load_inputs(cfg)
    hubs = add_pin_candidates(pins, hubs, cfg)
    dist_hp = distances(hubs, pins)
    dist_hh = hubhub_dist(hubs)

    elite, iter_records, best_obj = capsa(dist_hp, dist_hh, pins, hubs, cfg)

    open_mask = elite["open_mask"]; chosen_h = elite["chosen_h"]; chosen_d = elite["chosen_d"]; vol = elite["vol"]
    hubs_out = hubs.copy(); hubs_out["open"] = open_mask; hubs_out["assigned_kg"] = vol
    hubs_sel = hubs_out[hubs_out.open].sort_values("assigned_kg", ascending=False)

    # --- Merge hubs within 5 km ---
    hubs_sel, replacements_close = merge_hubs(hubs_sel, cfg)

    # --- DC Allocation ---
    dc_lat = dcs["lat"].to_numpy()
    dc_lon = dcs["lon"].to_numpy()
    dc_ids = dcs["dc_id"].astype(str).to_numpy()
    dc_names = dcs["dc_name"].to_numpy()

    hub_dc_id, hub_dc_name, hub_dc_dist = [], [], []
    for _, row in hubs_sel.iterrows():
        dists = haversine_km(row["lat"], row["lon"], dc_lat, dc_lon)
        j = np.argmin(dists)
        hub_dc_id.append(dc_ids[j]); hub_dc_name.append(dc_names[j]); hub_dc_dist.append(float(dists[j]))
    hubs_sel["nearest_dc_id"] = hub_dc_id
    hubs_sel["nearest_dc_name"] = hub_dc_name
    hubs_sel["nearest_dc_dist_km"] = hub_dc_dist

    # --- SAFE ASSIGNMENTS ---
    valid = chosen_h >= 0
    assign = pd.DataFrame({
        "pincode": pins["pincode"].to_numpy()[valid],
        "hub_id": hubs.loc[chosen_h[valid], "hub_id"].to_numpy(),
        "hub_name": hubs.loc[chosen_h[valid], "name"].to_numpy(),
        "distance_km": chosen_d[valid],
        "weight_kg": pins["weight_kg"].to_numpy()[valid]
    })
    hub_to_dc = hubs_sel.set_index("hub_id")[["nearest_dc_id","nearest_dc_name","nearest_dc_dist_km"]]
    assign = assign.merge(hub_to_dc, on="hub_id", how="left")

    unassigned = pins.loc[~valid, ["pincode","lat","lon","weight_kg"]].copy()
    unassigned["reason"] = f"No hub within {cfg['D_MAX']} km"

    # --- Hub Replacement Mapping (old hubs vs new hubs) ---
    rep_records = []
    if not hubs_sel.empty:
        for _, old in old_hubs.iterrows():
            dists = haversine_km(old["lat"], old["lon"], hubs_sel["lat"], hubs_sel["lon"])
            j = int(np.argmin(dists))
            rep_records.append({
                "old_hub_id": old["hub_id"],
                "old_hub_name": old["name"],
                "new_hub_id": hubs_sel.iloc[j]["hub_id"],
                "new_hub_name": hubs_sel.iloc[j]["name"],
                "replacement_distance_km": float(dists[j])
            })
    replacements_old_new = pd.DataFrame(rep_records)

    # --- Summary ---
    summary = {
        "pct_weight_le_30km": elite.get("pct30"),
        "pct_weight_le_50km": elite.get("pct50"),
        "hubs_open": int(len(hubs_sel)),
        "total_weight_kg": float(np.sum(pins.weight_kg)),
        "unassigned_pins": int(unassigned.shape[0]),
        "unassigned_weight_kg": float(unassigned["weight_kg"].sum()),
        "iterations_run": int(cfg["iters"]),
        "best_obj": float(best_obj)
    }

    # --- OUTPUTS IN EXCEL ---
    out_path = os.path.join(cfg["out_dir"], "capsa_service_output.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        hubs_sel.to_excel(writer, sheet_name="hubs_selected", index=False)
        assign.to_excel(writer, sheet_name="pin_assignment", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        unassigned.to_excel(writer, sheet_name="unassigned_pins", index=False)
        replacements_old_new.to_excel(writer, sheet_name="hub_replacements_old_new", index=False)
        replacements_close.to_excel(writer, sheet_name="hub_merges_within5km", index=False)
        # iterations:
        if iter_records:
            pd.DataFrame(iter_records).to_excel(writer, sheet_name="iterations", index=False)
        else:
            pd.DataFrame([], columns=["iteration","obj","best_obj_so_far","pct30","pct50","hubs_open"]).to_excel(writer, sheet_name="iterations", index=False)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Results written to: {out_path}")

if __name__ == "__main__":
    main()
