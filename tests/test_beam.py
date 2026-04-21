from beam import EulerBernoulliBeam

# TODO: test first beam natural frequencies against know
# TODO: test mode shape = +/- 2 at x=L
# TODO: test mode shape = 0 at x=0
# TODO: test mode orthogonality; M-K diagonal
# TODO: test phi^2 integral = L

def make_beam():
	b = 1e-2
	h = 2e-2
	mu = 1.0
	E = 210e9
	I = b * h ** 3 / 12
	L = 10.0
	return EulerBernoulliBeam(mu, E, I, L)
