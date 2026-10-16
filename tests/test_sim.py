import numpy as np


def test_reset_places_objects_where_asked(env):
    obs = env.reset(start=[0.10, 0.10], goal=[0.20, 0.30], obstacles=np.array([[0.15, 0.20, 0.0085]]))
    assert np.allclose(obs.cylinder, [0.10, 0.10], atol=0.003)
    assert np.allclose(obs.goal, [0.20, 0.30]) and obs.obstacles.shape == (1, 3)
    assert not obs.collision and not obs.success


def test_straight_push_moves_the_cap(env):
    obs = env.reset(start=[0.14, 0.20], goal=[0.14, 0.33], n_obstacles=0)
    env.target = np.array([0.14, 0.05]); env._settle(env.target)
    env.target = np.array([0.14, 0.165]); env._settle(env.target)
    y0 = env.observe().cylinder[1]
    for _ in range(8):
        obs = env.step([0.0, 0.01])
    assert obs.cylinder[1] - y0 > 0.04          # the cap was pushed along +y


def test_pushing_into_an_obstacle_flags_a_collision(env):
    env.reset(start=[0.08, 0.25], goal=[0.26, 0.25], obstacles=np.array([[0.12, 0.25, 0.0085]]))
    env.target = np.array([0.03, 0.25]); env._settle(env.target); env._collided = False
    flags = [env.step([0.01, 0.0]).collision for _ in range(10)]
    assert any(flags)


def test_layouts_respect_their_rules(env):
    for seed in range(15):
        env.rng = np.random.default_rng(seed)
        o = env.reset(opposite_sides=True)
        d = np.abs(o.goal - o.cylinder) / env.size
        assert 4 <= len(o.obstacles) <= 6 and d.max() > 0.5          # opposite edge zones along one axis
        o = env.reset(obstacle_on_line=True)
        assert len(o.obstacles) >= 3
