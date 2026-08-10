# Cycling benchmark continuation seed

`legacy-resistive-0p22-warmup.npz` is the two-cycle collocation cache selected
by the reference cycling run logged at a signed crank torque of `+0.22 N.m`
(resistive convention). It predates cache metadata, so this provenance cannot
be verified from the file alone and the source torque is declared explicitly
by the benchmark command.

The seed is not presented as a feasible solution of the target problem. It is
only a common primal continuation point for all backends. Before the solver
matrix starts, the workflow requires a one-RHO IPOPT solve at the
workflow-selected signed torque to converge and pass the physical checks. The
current endurance campaign uses `0.00 N.m`; older assisted campaigns used
`-0.20 N.m`.

The certified output is immutable within one workflow run and is shared by
every solver job in that run. It is not bitwise invariant across independent
runs: the non-convex IPOPT preparation can select a different stimulation
branch. Consequently, MadNLP performs one solver-specific periodic IPOPT
refinement on the exact MadNLP transcription before its timed RHO loop. This
preparation is reported separately and does not rebuild the graph during the
RHO sequence.

This platform dependence is now measured. Run `31419405169`, whose seed was
prepared on an Intel Xeon 8370C, produced a different `common-reduced.npz` from
runs `31420496210` and `31422321005`, both prepared on AMD EPYC 7763. The two
AMD files are bitwise identical. Intel versus AMD changes the biceps PW by up
to `180.6 us` in the seed and selects a different ACADOS branch: no recovery
through 150 RHO for the Intel seed, but two recoveries at RHO 5 in both AMD
runs. The workflow input `acados_seed_source_run_id` can temporarily pin an
artifact from a specified run and records all seed SHA-256 values. Because CI
artifacts expire, this is an ablation mechanism, not the durable solution; the
selected seed must ultimately be stored as a versioned benchmark asset.

The certified common seeds produced from this continuation may enforce start
constraints. Such a seed is accepted by a consumer that releases these
constraints because it belongs to a stricter feasible subset. Reuse in the
opposite direction remains forbidden: a seed built without start constraints
cannot silently initialize a consumer that requires them.

SHA-256:

`eb150a08e936df019bffe918bf8b38586aa664f67b6aa0b08529e1deccd3083e`
