The internal dynamics of decisions -- Nathan M. Thornhill (ICSAC Institute)
Supplementary material: analysis code, results, and frozen pre-registrations.

This deposit is a single tree. src/ holds the analysis code and the shared instrument
src/demix.py; results/ holds the result files underlying every figure and statistic;
figures/ holds the figure-generating code; data/ holds the construction-task item set.

The two pre-registered controls are cmc/ (computation-matched architecture control) and
realmodel/ (real-model across-token generation test). Each carries its own FREEZE.txt
SHA-256 manifest, and the cmc freeze references src/demix.py as its shared instrument.
Each pre-registration fixed its design, metric and kill conditions before any result was
computed. To verify a freeze, extract this archive and recompute SHA-256 of each listed
file from the resulting cross-substrate-dynamics/ folder.
