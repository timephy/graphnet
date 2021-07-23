# ~/.local/bin/i3cols extr_sep \
# ~/.local/bin/i3cols extr_sep   /remote/ceph/user/h/haminh/GraphNet/input/flat/*.zst \
~/.local/bin/i3cols extr_sep  /remote/ceph/user/h/haminh/GraphNet/input/l7/160000/i3/*.zst \
    --outdir /remote/ceph/user/h/haminh/GraphNet/input/l7/160000/i3cols/ \
    --procs 50 \
    --concatenate-and-index-by subrun \
    --keys I3EventHeader \
            L7_MuonClassifier_FullSky_ProbNu \
            L7_MuonClassifier_Upgoing_ProbNu \
            L7_CoincidentMuon_Variables \
            L4_NoiseClassifier_ProbNu \
            L5_SANTA_DirectPulsesHitMultiplicity \
            I3MCTree \
            MCInIcePrimary \
            InIcePulses \
            SplitInIcePulsesSRT \
            SplitInIcePulsesTWSRT \
            SRTTWOfflinePulsesDC \
            I3MCWeightDict \
            MCDeepCoreStartingEvent \
            I3MCWeightDict \
            L7_MuonClassifier_ProbNu \
            L7_CoincidentMuon_bool \
            L7_oscNext_bool \
            L7_reconstructed_zenith \
            L7_reconstructed_total_energy \
            L7_reconstructed_vertex_x \
            L7_reconstructed_vertex_y \
            L7_reconstructed_vertex_z \
            L7_PIDClassifier_ProbTrack
