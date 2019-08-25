from sac import SAC

if __name__ == "__main__":
    sac = SAC(env_name="Humanoid-v2",
              data_save_dir="../Humanoid-v2")
    sac.train(resume_training=True)
    sac.test(render=True, use_internal_policy=False, num_games=10)
