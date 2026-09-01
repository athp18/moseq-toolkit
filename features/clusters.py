_REQUIRED_COLUMNS = {
    'syllable',
    'mean_velocity',
    'mean_displacement',
    'n_frames'
}


def _coverage_filter(syllable_kinematics, coverage):
    """Keep the most-frequent syllables whose cumulative frame count covers
    `coverage` fraction of all frames; always keeps at least one syllable."""
    if not (0 < coverage <= 1):
        raise ValueError(f'coverage must be in (0, 1], got {coverage}')

    ordered = syllable_kinematics.sort_values('n_frames', ascending=False)
    total_frames = ordered['n_frames'].sum()
    if total_frames <= 0:
        return ordered.iloc[0:0]

    cumulative_frac = ordered['n_frames'].cumsum() / total_frames
    # include the syllable that crosses the coverage threshold
    n_keep = max(1, int((cumulative_frac < coverage).sum()) + 1)
    n_keep = min(n_keep, len(ordered))
    return ordered.iloc[:n_keep]


def assign_clusters(syllable_kinematics, min_frames=5, coverage=None):
    """Assign each syllable to one of four clusters (motor/cognitive x
    improvement/deficit) via a nested median split on velocity and
    displacement. `coverage`, if given, overrides `min_frames` as the
    inclusion filter. Returns a dict mapping syllable -> cluster label."""
    missing = _REQUIRED_COLUMNS - set(syllable_kinematics.columns)
    if missing:
        raise ValueError(f'Missing {missing} columns')

    if coverage is not None:
        syllable_kinematics = _coverage_filter(syllable_kinematics, coverage)
    else:
        syllable_kinematics = syllable_kinematics[syllable_kinematics['n_frames'] >= min_frames]

    # nested split: velocity median is computed within each displacement
    # group separately, so all four clusters get populated even when
    # velocity and displacement are correlated
    med_disp = syllable_kinematics["mean_displacement"].median()
    high_disp_mask = syllable_kinematics["mean_displacement"] >= med_disp

    med_vel_high_disp = syllable_kinematics.loc[high_disp_mask, "mean_velocity"].median()
    med_vel_low_disp = syllable_kinematics.loc[~high_disp_mask, "mean_velocity"].median()

    def classify_cluster(row):
        high_disp = row["mean_displacement"] >= med_disp
        med_vel = med_vel_high_disp if high_disp else med_vel_low_disp
        high_vel = row["mean_velocity"] >= med_vel

        if high_vel and high_disp:
            return "motor improvement"
        elif not high_vel and high_disp:
            return "motor deficit"
        elif high_vel and not high_disp:
            return "cognitive improvement"
        else:
            return "cognitive deficit"

    syllable_kinematics["cluster"] = syllable_kinematics.apply(classify_cluster, axis=1)
    return dict(zip(syllable_kinematics["syllable"], syllable_kinematics["cluster"]))
