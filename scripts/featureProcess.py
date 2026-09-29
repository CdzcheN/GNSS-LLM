import json
import os
import pandas as pd
import numpy as np


def _to_slots(arr, n=32):
    a = np.asarray(arr, dtype=float)
    if a.size != n:
        print(f"  WARNING: slot length {a.size} != {n}, padded/truncated")
        out = np.zeros(n)
        k = min(a.size, n)
        out[:k] = a[:k]
        return out
    return a


def _column_order():
    pvt_cols = ['hAcc', 'vAcc', 'tAcc', 'clkB', 'clkD', 'gSpeed', 'pDOP', 'tDOP', 'hDOP']
    obs_cols = ['NumSats', 'AvgCNO', 'MaxRes']
    sat_cols = []
    for prn in range(1, 33):
        sat_cols.extend([f'CNO_G{prn:02d}', f'Res_G{prn:02d}', f'Elev_G{prn:02d}'])
    return ['Timestamp', 'Day', 'Hour'] + pvt_cols + obs_cols + sat_cols + ['Label']


def _process_day(root_path, day, hours_list, final_order, output_file):
    """处理一天的所有小时，逐小时追加到 output_file，返回本天有效行数。"""
    day_path = os.path.join(root_path, str(day))
    day_rows = 0

    for h in hours_list:
        obs_path = os.path.join(day_path, f'observation{h}.json')
        sat_path = os.path.join(day_path, f'satelliteInfomation{h}.json')
        pvt_path = os.path.join(day_path, f'pvtSolution{h}.json')

        if not (os.path.exists(obs_path) and os.path.exists(sat_path) and os.path.exists(pvt_path)):
            continue

        print(f"Processing: Day {day} | Hour {h}")
        try:
            with open(obs_path, 'r', encoding='utf-8') as f1, \
                 open(sat_path, 'r', encoding='utf-8') as f2, \
                 open(pvt_path, 'r', encoding='utf-8') as f3:
                obs_raw = json.load(f1)
                sat_raw = json.load(f2)
                pvt_raw = json.load(f3)
        except Exception as e:
            print(f" Error reading {day}/{h}: {e}")
            continue

        def _ts_index(ts_list, name):
            seen, dups = set(), set()
            for t in ts_list:
                if t in seen:
                    dups.add(t)
                seen.add(t)
            if dups:
                print(f"  WARNING [{name}]: {len(dups)} duplicated timestamp(s), e.g. {sorted(dups)[:3]}")
            return {t: i for i, t in enumerate(ts_list)}

        obs_dict = _ts_index(obs_raw['recordTime'], 'observation')
        sat_dict = _ts_index(sat_raw['recordTime'], 'satelliteInfomation')
        pvt_dict = _ts_index(pvt_raw['recordTime'], 'pvtSolution')

        common_ts = sorted(set(obs_dict) & set(sat_dict) & set(pvt_dict))
        union_ts = sorted(set(obs_dict) | set(sat_dict) | set(pvt_dict))
        print(f"  Timestamps: obs={len(obs_dict)} sat={len(sat_dict)} pvt={len(pvt_dict)} "
              f"aligned={len(common_ts)} union={len(union_ts)} missing={len(union_ts) - len(common_ts)}")

        hour_rows = []

        for t_str in common_ts:
            i_obs, i_sat, i_pvt = obs_dict[t_str], sat_dict[t_str], pvt_dict[t_str]
            row = {'Timestamp': t_str, 'Day': day, 'Hour': int(t_str[11:13])}

            def get_pvt_val(key, index, default=np.nan):
                if key in pvt_raw and index < len(pvt_raw[key]):
                    return pvt_raw[key][index]
                return default

            row['hAcc'] = get_pvt_val('hAcc', i_pvt)
            row['vAcc'] = get_pvt_val('vAcc', i_pvt)
            row['tAcc'] = get_pvt_val('tAcc', i_pvt)
            row['clkB'] = get_pvt_val('clkB', i_pvt)
            row['clkD'] = get_pvt_val('clkD', i_pvt)
            row['gSpeed'] = get_pvt_val('gSpeed', i_pvt)
            row['pDOP'] = get_pvt_val('pDOP', i_pvt)
            row['tDOP'] = get_pvt_val('tDOP', i_pvt)
            row['hDOP'] = get_pvt_val('hDOP', i_pvt)

            cno_list = _to_slots(obs_raw['cn0_G1'][i_obs])
            res_list = _to_slots(sat_raw['prRes_G'][i_sat])
            elv_list = _to_slots(sat_raw['elev_G'][i_sat])

            tracked = cno_list > 0.5
            valid_cno = cno_list[tracked]
            valid_res = np.abs(res_list[np.abs(res_list) > 0.001])

            row['NumSats'] = len(valid_cno)
            row['AvgCNO'] = np.mean(valid_cno) if len(valid_cno) > 0 else 0
            row['MaxRes'] = np.max(valid_res) if len(valid_res) > 0 else 0

            for prn in range(1, 33):
                idx = prn - 1
                row[f'CNO_G{prn:02d}'] = cno_list[idx] if tracked[idx] else 0
                row[f'Res_G{prn:02d}'] = res_list[idx] if tracked[idx] and np.abs(res_list[idx]) > 0.001 else 0
                row[f'Elev_G{prn:02d}'] = elv_list[idx] if tracked[idx] else 0

            row['Label'] = 0
            hour_rows.append(row)

        if hour_rows:
            pd.DataFrame(hour_rows)[final_order].to_csv(
                output_file, mode='a', header=False, index=False, encoding='utf-8'
            )
            day_rows += len(hour_rows)
            print(f"    -> appended {len(hour_rows)} rows (day {day} subtotal={day_rows})")

        del hour_rows, obs_raw, sat_raw, pvt_raw

    return day_rows


def extract_integrated_features_by_group(root_path, days_list, hours_list,
                                        output_prefix='gnss_features',
                                        group_size=5, output_dir='.'):
    """
    每 group_size 天合并到一个 CSV 文件。
    例如 days_list=['12'..'20'], group_size=5 →
        gnss_features_12-16.csv   (12,13,14,15,16)
        gnss_features_17-20.csv   (17,18,19,20)
    """
    root_path = root_path.strip()
    final_order = _column_order()
    os.makedirs(output_dir, exist_ok=True)

    days_list = list(days_list)
    n = len(days_list)
    total_rows_all = 0
    produced_files = []

    for start in range(0, n, group_size):
        group_days = days_list[start:start + group_size]
        first_day, last_day = group_days[0], group_days[-1]
        out_name = f"{output_prefix}_{first_day}-{last_day}.csv"
        out_path = os.path.join(output_dir, out_name)

        print("\n" + "=" * 70)
        print(f"GROUP: days {group_days} -> {out_path}")
        print("=" * 70)

        # 每个组新建文件并写表头
        if os.path.exists(out_path):
            os.remove(out_path)
        pd.DataFrame(columns=final_order).to_csv(
            out_path, index=False, encoding='utf-8'
        )

        group_rows = 0
        for day in group_days:
            day_rows = _process_day(root_path, day, hours_list, final_order, out_path)
            group_rows += day_rows

        print(f"[GROUP DONE] {out_name} | rows={group_rows}")
        produced_files.append((out_name, group_rows))
        total_rows_all += group_rows

    print("\n" + "=" * 70)
    print(" ALL GROUPS COMPLETE")
    for name, rows in produced_files:
        print(f"  {name}  rows={rows}")
    print(f"  TOTAL rows = {total_rows_all}")
    print("=" * 70)


if __name__ == "__main__":
    extract_integrated_features_by_group(
        root_path=r'/home/chen/文档/VsCodeDoc/GNSS-DATA/GNSS Dataset (with Interference and Spoofing) Part II/processed data',
        days_list=['21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
        hours_list=range(24),
        output_prefix='gnss_complete_featuresObsSatPvt',
        group_size=5,
        output_dir='.',
    )
