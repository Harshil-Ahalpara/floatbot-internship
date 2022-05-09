from happytransformer import TTTrainArgs
training_args = TTTrainArgs(batch_size=6,
                            num_train_epochs=3)


from happytransformer import TTSettings
testing_args = TTSettings(num_beams=5, 
                          min_length=1)
