from src.io_utils import atomic_write_csv, atomic_write_json
from src.cosmed import load_cosmed_take
from src.estimator.cotracker_rf import estimate_rf_from_cotracker
from src.signals.quality import signal_quality_metrics, corrcoef_safe
from src.signals.plots import save_overlay_plot, save_coherence_plot, save_waveform_plot
from src.signals.preprocess import align_video_waveform_to_ref
from src.signals.events import breath_times_from_rf, match_events_nearest
from src.vt_calib.linear import fit_linear_calibration
from src.vt_calib.blocked_cv import cv_calibrate_linear_blocked
import os
import datetime as dt
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline.specs import ExperimentSpec


class BreathingPipeline:
    def __init__(self, base_results_dir="Results", report_name="VT_Report_LIVE.csv"):
        """
        Creates the base results directory if needed and sets the path of the CSV report where runs will be appended incrementally.
        """
        self.base_results = base_results_dir
        os.makedirs(self.base_results, exist_ok=True)
        self.report_path = os.path.join(self.base_results, report_name)

    def _make_run_id(self, spec: ExperimentSpec, roi: str, use_pca: bool) -> str:
        """
        Build a unique run identifier for a given experiment configuration. The run_id encodes camera/subject/take/ROI and whether PCA-based motion extraction is used, and is used for result folder naming + report logging.
        """
        pca_str = "PCA" if use_pca else "NoPCA"
        return f"{spec.camera}_{spec.subject}_{spec.marker}_{roi}_{pca_str}"

    def _done_flag_path(self, out_dir: str) -> str:
        """
        Return the filesystem path of the run completion marker file for a run folder.
        """
        return os.path.join(out_dir, "_DONE.json")

    def _load_done_run_ids_from_report(self) -> set:
        """
        Load run_ids already present in the live CSV report (for resume and skip logic).
        """
        if not os.path.exists(self.report_path):
            return set()
        try:
            df = pd.read_csv(self.report_path)
            if "run_id" in df.columns:
                return set(df["run_id"].astype(str).tolist())
            # fallback if old report without run_id
            needed = {"Camera", "Subject", "Take", "ROI", "PCA"}
            if needed.issubset(df.columns):
                return set(
                    (df["Camera"].astype(str) + "_" +
                     df["Subject"].astype(str) + "_" +
                     df["Take"].astype(str) + "_" +
                     df["ROI"].astype(str) + "_" +
                     df["PCA"].astype(str)).tolist()
                )
        except Exception:
            pass
        return set()

    def _is_run_done(self, out_dir: str, run_id: str, done_run_ids: set) -> bool:
        """
        Determine whether a run should be skipped because it is already complete or not.

        A run is considered done if any of the following is true:
            - a _DONE.json flag exists in the output directory
            - a results_ts.csv file already exists (time-series outputs saved)
            - its run_id is already present in the live report CSV
        """
        if os.path.exists(self._done_flag_path(out_dir)):
            return True
        if os.path.exists(os.path.join(out_dir, "results_ts.csv")):
            return True
        if run_id in done_run_ids:
            return True
        return False

    def _mark_run_done(self, out_dir: str, run_id: str, record: dict):
        """
        Write a completion marker (_DONE.json) containing minimal run metadata.
        """
        atomic_write_json(
            {
                "run_id": run_id,
                "when": dt.datetime.now().isoformat(),
                "record": {k: record.get(k, None) for k in ["Camera", "Subject", "Take", "ROI", "PCA", "FPS"]},
            },
            self._done_flag_path(out_dir),
        )

    def _append_report_row(self, record: dict):
        """
        Append a single run summary row to the live CSV report.
        """
        cols = [
            "run_id",
            "Camera", "Subject", "Take", "ROI", "PCA", "FPS",

            "Corr_VT", "N_Samples_VT",

            "VT_cal_a", "VT_cal_b", "VT_cal_train_frac",
            "VT_MAE_test_L", "VT_RMSE_test_L", "VT_R2_test",

            "VT_corr_z", "VT_best_lag_s", "VT_coh_mean_band", "VT_coh_peak_band",
            "VE_corr_z", "VE_best_lag_s", "VE_coh_mean_band", "VE_coh_peak_band",

            "RR_gt_mean_bpm", "RR_vid_psd_bpm", "RR_error_bpm", "RR_abs_error_bpm",
        ]

        row = {k: record.get(k, np.nan) for k in cols}
        df_row = pd.DataFrame([row])

        file_exists = os.path.exists(self.report_path)
        df_row.to_csv(self.report_path, mode="a", header=not file_exists, index=False)

        # ensure it’s flushed to disk
        try:
            with open(self.report_path, "a") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def save_summary_report_snapshot(self):
        """
        Save a timestamped snapshot of the live CSV report and generate summary charts.
        """
        if not os.path.exists(self.report_path):
            print("No live report found; nothing to snapshot.")
            return

        try:
            df = pd.read_csv(self.report_path)
        except Exception as e:
            print("Could not read live report:", e)
            return

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(self.base_results, f"VT_Report_SNAPSHOT_{timestamp}.csv")
        df.to_csv(snap_path, index=False)
        print(f"\n--- Snapshot saved to: {snap_path} ---")

        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="ROI", y="VT_MAE_test_L", hue="PCA", errorbar=None)
            plt.title("VT Calibration Test MAE (Liters) by ROI and Method")
            plt.ylabel("MAE (L) ↓")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_results, f"Summary_VT_MAE_Chart_{timestamp}.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="ROI", y="VT_R2_test", hue="PCA", errorbar=None)
            plt.title("VT Calibration Test R² by ROI and Method")
            plt.ylabel("R² ↑")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_results, f"Summary_VT_R2_Chart_{timestamp}.png"))
            plt.close()
        except Exception as e:
            print("Could not create summary chart:", e)

    def run_batch(self, frame_root_base, experiment_specs, rois, pcas, lo_hz, hi_hz):
        """
        Run the full breathing evaluation pipeline over multiple experiments and settings.

        For every combination (experiment spec × ROI × PCA/non-PCA), this method:
            - loads COSMED ground-truth signals (Rf, VT, VE)
            - runs the CoTracker-based respiratory-rate estimator on the video frames
            - saves raw/filtered waveforms and diagnostic plots (overlay/coherence)
            - estimates best lag between video waveform and COSMED reference waveform
            - performs breath-by-breath VT proxy matching and blocked-CV linear calibration
            - writes per-run outputs in a structured results folder
            - appends a run summary row to the live report CSV
            - marks runs complete using a "_DONE.json" flag for resume safety
        Runs are skipped if already completed (DONE flag, existing results_ts.csv, or run_id
        already present in the report) so the pipeline can resume after interruptions without
        recomputing everything.

        Args:
            frame_root_base (str): Root directory containing frame folders organized by camera.
            experiment_specs (list[ExperimentSpec]): List of experiment specifications to run.
            rois (list[str]): ROI modes/regions to evaluate (["chest", "abdomen", "segmented"]).
            pcas (list[bool]): Whether to run PCA extraction or not ([False, True]).
            lo_hz (float): Lower frequency bound (Hz) for the breathing band.
            hi_hz (float): Upper frequency bound (Hz) for the breathing band.

        Returns:
            None
        """
        done_run_ids = self._load_done_run_ids_from_report()

        total_runs = len(experiment_specs) * len(rois) * len(pcas)
        count = 0

        print(f"--- Starting/Resuming Batch: {total_runs} total runs | LIVE report: {self.report_path} ---")

        for spec in experiment_specs:
            frame_dir = os.path.join(frame_root_base, spec.camera, spec.folder)

            if not os.path.exists(frame_dir):
                print(f"SKIP: Missing folder {frame_dir}")
                continue

            try:
                gt_full = load_cosmed_take(spec.cosmed_path, spec.marker, resample_dt_s=0.25)
                png_count = len(glob.glob(os.path.join(frame_dir, "*.png")))
                if png_count == 0:
                    print(f"SKIP: No PNGs in {frame_dir}")
                    continue
                vid_duration = (png_count - 1) / spec.fps
                gt = gt_full[gt_full["time_s"] <= vid_duration].reset_index(drop=True)
            except Exception as e:
                print(f"SKIP: GT Load failed for {spec.marker} ({spec.camera}) - {e}")
                continue

            for roi in rois:
                for use_pca in pcas:
                    count += 1
                    run_id = self._make_run_id(spec, roi, use_pca)
                    pca_str = "PCA" if use_pca else "NoPCA"

                    out_dir = os.path.join(self.base_results, spec.camera, spec.subject, roi, pca_str, spec.marker)
                    os.makedirs(out_dir, exist_ok=True)

                    if self._is_run_done(out_dir, run_id, done_run_ids):
                        print(f"[{count}/{total_runs}] SKIP (done): {run_id}")
                        continue

                    print(f"[{count}/{total_runs}] Processing: {run_id} (FPS: {spec.fps})...")

                    try:
                        rr_video_bpm, sig_filtered, sig_raw, t_amp, amp_video = estimate_rf_from_cotracker(frame_dir, spec.camera, spec.fps, roi_region=roi, with_pca=use_pca, lo_hz=lo_hz, hi_hz=hi_hz)

                        record = {
                            "run_id": run_id,
                            "Camera": spec.camera, "Subject": spec.subject, "Take": spec.marker,
                            "ROI": roi, "PCA": use_pca, "FPS": spec.fps,

                            "Corr_VT": np.nan, "N_Samples_VT": 0,

                            "VT_cal_a": np.nan, "VT_cal_b": np.nan, "VT_cal_train_frac": np.nan,
                            "VT_MAE_test_L": np.nan, "VT_RMSE_test_L": np.nan, "VT_R2_test": np.nan,

                            "VT_corr_z": np.nan, "VT_best_lag_s": np.nan, "VT_coh_mean_band": np.nan, "VT_coh_peak_band": np.nan,
                            "VE_corr_z": np.nan, "VE_best_lag_s": np.nan, "VE_coh_mean_band": np.nan, "VE_coh_peak_band": np.nan,

                            "RR_gt_mean_bpm": np.nan,
                            "RR_vid_psd_bpm": float(rr_video_bpm) if np.isfinite(rr_video_bpm) else np.nan,
                            "RR_error_bpm": np.nan,
                            "RR_abs_error_bpm": np.nan,
                        }

                        if len(sig_raw) > 0:
                            wave_t = np.arange(len(sig_raw)) / spec.fps
                            df_wave = pd.DataFrame({
                                "time_s": wave_t,
                                "raw_displacement": sig_raw,
                                "filtered_signal": sig_filtered
                            })
                            atomic_write_csv(df_wave, os.path.join(out_dir, "breathing_waveform.csv"))
                            save_waveform_plot(wave_t, sig_raw, sig_filtered, out_dir, run_id)

                        df_run = gt.copy()

                        burn_s = 2.0
                        rr_gt_mean = float(np.nanmean(df_run.loc[df_run["time_s"] >= burn_s, "Rf"].to_numpy(dtype=float)))
                        record["RR_gt_mean_bpm"] = rr_gt_mean

                        if np.isfinite(record["RR_gt_mean_bpm"]) and np.isfinite(record["RR_vid_psd_bpm"]):
                            record["RR_error_bpm"] = record["RR_vid_psd_bpm"] - record["RR_gt_mean_bpm"]
                            record["RR_abs_error_bpm"] = abs(record["RR_error_bpm"])

                        lag_s = 0.0

                        if len(sig_filtered) > 32 and len(df_run) > 32:
                            dt_ref = float(np.median(np.diff(df_run["time_s"].to_numpy())))
                            fs_ref = 1.0 / dt_ref if dt_ref > 1e-9 else 1.0

                            t_video = np.arange(len(sig_filtered)) / spec.fps
                            df_run_time = df_run["time_s"].to_numpy(dtype=float)

                            vid_unshift = np.interp(df_run_time, t_video, sig_filtered, left=sig_filtered[0], right=sig_filtered[-1])

                            vt_ref = df_run["VT"].to_numpy(dtype=float)
                            ve_ref = df_run["VE"].to_numpy(dtype=float)

                            m_vt_raw = signal_quality_metrics(vt_ref, vid_unshift, fs=fs_ref, lo_hz=0.07, hi_hz=1.0)
                            lag_s = m_vt_raw["best_lag_s"]
                            if not np.isfinite(lag_s) or abs(lag_s) > 2.0:
                                lag_s = 0.0

                            record["VT_best_lag_s"] = lag_s
                            record["VE_best_lag_s"] = lag_s

                            vid_aligned = align_video_waveform_to_ref(df_run_time, t_video, sig_filtered, lag_s)
                            df_run["Video_waveform"] = vid_aligned

                            m_vt = signal_quality_metrics(vt_ref, vid_aligned, fs=fs_ref, lo_hz=0.07, hi_hz=1.0, forced_lag_s=0.0)
                            m_ve = signal_quality_metrics(ve_ref, vid_aligned, fs=fs_ref, lo_hz=0.07, hi_hz=1.0, forced_lag_s=0.0)
                            
                            record.update({
                                "VT_corr_z": m_vt["corr_z"],
                                "VT_coh_mean_band": m_vt["coh_mean_band"],
                                "VT_coh_peak_band": m_vt["coh_peak_band"],
                                "VE_corr_z": m_ve["corr_z"],
                                "VE_coh_mean_band": m_ve["coh_mean_band"],
                                "VE_coh_peak_band": m_ve["coh_peak_band"],
                            })

                            save_overlay_plot(
                                t=df_run["time_s"].to_numpy(),
                                vid_z=m_vt["vid_z"], ref_z=m_vt["ref_z"],
                                out_path=os.path.join(out_dir, f"{run_id}_overlay_VT.png"),
                                title=(f"{run_id} | VT vs Video(z) | corr={m_vt['corr_z']:.3f} | applied_lag={lag_s:.2f}s | "
                                       f"coh_mean={m_vt['coh_mean_band']:.3f} | coh_peak={m_vt['coh_peak_band']:.3f}")
                            )

                            save_overlay_plot(
                                t=df_run["time_s"].to_numpy(),
                                vid_z=m_ve["vid_z"], ref_z=m_ve["ref_z"],
                                out_path=os.path.join(out_dir, f"{run_id}_overlay_VE.png"),
                                title=(f"{run_id} | VE vs Video(z) | corr={m_ve['corr_z']:.3f} | applied_lag={lag_s:.2f}s | "
                                       f"coh_mean={m_ve['coh_mean_band']:.3f} | coh_peak={m_ve['coh_peak_band']:.3f}")
                            )

                            if len(m_vt["coh_f"]) > 0:
                                save_coherence_plot(m_vt["coh_f"], m_vt["coh_Cxy"],
                                                    os.path.join(out_dir, f"{run_id}_coherence_VT.png"),
                                                    title=f"{run_id} | Coherence (VT vs Video z)", lo_hz=0.07, hi_hz=1.0)
                            if len(m_ve["coh_f"]) > 0:
                                save_coherence_plot(m_ve["coh_f"], m_ve["coh_Cxy"],
                                                    os.path.join(out_dir, f"{run_id}_coherence_VE.png"),
                                                    title=f"{run_id} | Coherence (VE vs Video z)", lo_hz=0.07, hi_hz=1.0)

                        if len(t_amp) > 5:
                            lag_s = float(np.clip(lag_s, -2.0, 2.0))
                        
                            t_breath_ref = breath_times_from_rf(df_run["time_s"].to_numpy(dtype=float),
                                                                df_run["Rf"].to_numpy(dtype=float))
                        
                            vt_breath = np.interp(
                                t_breath_ref,
                                df_run["time_s"].to_numpy(dtype=float),
                                df_run["VT"].to_numpy(dtype=float),
                                left=np.nan,
                                right=np.nan
                            )
                        
                            t_vid_evt = np.asarray(t_amp, float) + lag_s
                            a_vid_evt = np.asarray(amp_video, float)
                            
                            order = np.argsort(t_vid_evt)
                            t_vid_evt = t_vid_evt[order]
                            a_vid_evt = a_vid_evt[order]
                            
                            a_match = match_events_nearest(t_breath_ref, t_vid_evt, a_vid_evt, max_dt=2.0)

                            m = np.isfinite(a_match) & np.isfinite(vt_breath)
                            n_match = int(np.sum(m))
                            record["N_Samples_VT"] = n_match
                            record["Corr_VT"] = corrcoef_safe(vt_breath, a_match)
                            
                            if n_match >= 20:
                                cv = cv_calibrate_linear_blocked(a_match, vt_breath, k=5, min_train=20, min_test=5)
                                record["VT_MAE_test_L"] = cv["mae"]
                                record["VT_RMSE_test_L"] = cv["rmse"]
                                record["VT_R2_test"] = cv["r2"]
                                a_fit, b_fit = fit_linear_calibration(a_match[m], vt_breath[m])
                                record["VT_cal_a"] = a_fit
                                record["VT_cal_b"] = b_fit

                                vt_pred = a_fit * a_match[m] + b_fit
                        
                                # Plot: breath-level calibrated scatter
                                plt.figure(figsize=(6, 6))
                                plt.scatter(vt_breath[m], vt_pred, alpha=0.5)
                                mn = float(min(np.min(vt_breath[m]), np.min(vt_pred)))
                                mx = float(max(np.max(vt_breath[m]), np.max(vt_pred)))
                                plt.plot([mn, mx], [mn, mx], linewidth=2)
                                plt.title(f"Breath-level VT calibration (blocked CV)\n"
                                          f"CV MAE={cv['mae']:.3f}L | RMSE={cv['rmse']:.3f}L | R²={cv['r2']:.3f} | n={cv['n']}")
                                plt.xlabel("VT true (L)")
                                plt.ylabel("VT predicted (L)")
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()
                                plt.savefig(os.path.join(out_dir, f"{run_id}_VT_breath_calibrated.png"))
                                plt.close()
                        
                                # Plot: proxy vs VT (breath-level)
                                plt.figure(figsize=(6, 6))
                                sns.regplot(x=vt_breath[m], y=a_match[m], scatter_kws={'alpha':0.5})
                                plt.title(f"Breath-level VT vs Video Proxy\nr = {record['Corr_VT']:.2f} | n={record['N_Samples_VT']}")
                                plt.xlabel("VT true (L)")
                                plt.ylabel("Video amplitude proxy")
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()
                                plt.savefig(os.path.join(out_dir, f"{run_id}_VT_breath_proxy.png"))
                                plt.close()
                            else:
                                pass
                        
                        atomic_write_csv(df_run, os.path.join(out_dir, "results_ts.csv"))

                        self._append_report_row(record)
                        self._mark_run_done(out_dir, run_id, record)
                        done_run_ids.add(run_id)

                    except Exception as e:
                        print(f"!!! Error in {run_id}: {e}")
                        import traceback
                        traceback.print_exc()

        self.save_summary_report_snapshot()

