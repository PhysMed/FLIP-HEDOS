class Particle:
    def __init__(self, flag=-1, trajectory=None, postraj=None, posspace=None, dose=0, index=None):
        self.Flag = flag
        self.Trajectory = trajectory
        self.PosTraj = postraj
        self.PosSpace = posspace
        self.Dose = dose
        self.Index = index
        self.When_in =[]
        self.When_out = []


class SeveralDoseInfo:
    def __init__(self):
        self.NumSpots = None
        self.AngleGantry = None
        self.Field = None
        self.Segment = None
        self.Dose = None


class TrajectoryInDose:
    def __init__(self):
        self.i = []
        self.j = []
        self.k = []
