import evaluate_surfaces as e
e.R=e.R/'secondary'
if not (e.R/'cohort.json').exists():(e.R/'cohort.json').symlink_to(e.R.parent/'cohort.json')
e.main()
