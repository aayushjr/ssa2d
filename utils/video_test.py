import dataloader

params = {'batch_size': 1,
          'in_dim': (224, 224),
          'n_channels': 3,
          'n_frames': 16,
          'frame_steps': (5,),
          'out_dim': (56, 56),
          'n_actions': 0,
          'shuffle_seed': 1}

train_loader = dataloader.VidORDataloader(mode=dataloader.TRAIN, samples_per_class=1,
                                          shuffle=True, **params)
