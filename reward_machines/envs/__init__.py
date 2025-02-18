from gym.envs.registration import register

# ----------------------------------------- Half-Cheetah

# register(
#     id='Half-Cheetah-RM1-v0',
#     entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM1',
#     max_episode_steps=1000,
# )
# register(
#     id='Half-Cheetah-RM2-v0',
#     entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM2',
#     max_episode_steps=1000,
# )


# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0' % i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d' % i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0' % i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d' % i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='Office-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-remote-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnvRemote',
    max_episode_steps=1000
)

register(
    id='Office-single-v0',
    entry_point='envs.grids.grid_environment:OfficeRM3Env',
    max_episode_steps=1000
)

# ----------------------------------------- CRAFT
for i in range(11):
    w_id = 'Craft-M%d-v0' % i
    w_en = 'envs.grids.grid_environment:CraftRMEnvM%d' % i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

for i in range(11):
    w_id = 'Craft-single-M%d-v0' % i
    w_en = 'envs.grids.grid_environment:CraftRM10EnvM%d' % i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

# ----------------------------------------- ILG-Learn
register(
    id='diag3x3-sparse-v0',
    entry_point='envs.ilg_rm.ilg_environment:MyDiag3x3SparseEnv',
    max_episode_steps=600,
)

register(
    id='diag3x3-dense-v0',
    entry_point='envs.ilg_rm.ilg_environment:MyDiag3x3DenseEnv',
    max_episode_steps=600, # should have  been 800 to match rce 
)

register(
    id='diag5x5-dense-v0',
    entry_point='envs.ilg_rm.ilg_environment:MyDiag5x5DenseEnv',
    max_episode_steps=1600, # should have  been 800 to match rce 
)

register(
    id='diag7x7-dense-v0',
    #point_maze-7x7-diagonal-one-sparse-10goals
    entry_point='envs.ilg_rm.ilg_environment:MyDiag7x7DenseEnv',
    max_episode_steps=2400, # this matches rce but is a lot.
)


register(
    id='diag7x7-coarse-v0',
    #point_maze-7x7-diagonal-one-sparse-10goals
    entry_point='envs.ilg_rm.ilg_environment:MyDiag7x7CoarseEnv',
    max_episode_steps=2400, # this matches rce but is a lot.
)



register(
    id='stackChoiceOutwardview-dense-v0',
    entry_point='envs.ilg_rm.ilg_environment:MyStackChoiceOutwardviewEnv',
    max_episode_steps=250,
)


register(
    id='stackAB-dense-v0',
    entry_point='envs.ilg_rm.ilg_environment:MyStackABEnv',
    max_episode_steps=250,
)