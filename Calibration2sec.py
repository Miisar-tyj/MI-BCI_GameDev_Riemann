import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance
from sklearn.model_selection import cross_val_score

class BCICalibrator:
    def __init__(self, n_chan=18, n_samples = 625):
        self.cov_estimator = Covariances('scm')
        self.mdm = MDM(metric='riemann')
        self.empty_trials = np.empty((0, n_chan, n_samples))
        self.action_trials = np.empty((0, n_chan, n_samples))
        self.lefthand_trials = np.empty((0,n_chan,n_samples))
        self.righthand_trials = np.empty((0,n_chan,n_samples))
        self.feet_trials = np.empty((0,n_chan,n_samples))
        self.tongue_trials = np.empty((0,n_chan,n_samples))
       # self.calibration_data = {
        #                        'empty': np.empty((0, 18, 1125)),  # (n, 18, 1125) where n starts at 0
         #                       'action': np.empty((0, 18, 1125))
         #                       }            #tbh here just change to np.array (0:blah blah blha ==empty) (1:blahblahblbha == lefthand)
        self.covs_ref = None                                            #and dont use list
        self.mean_ref = None
        self.is_trained = False
        self.n_chan= n_chan
        self.n_samples = n_samples
        
    def add_calibration_data(self, trial_data, label):
        """Store raw EEG data for calibration"""
        if not isinstance(trial_data, np.ndarray):
              raise ValueError("trial_data must be a NumPy array")
        if trial_data.ndim != 2:
              raise ValueError("trial_data must be 2D (channels x samples)")
        if trial_data.shape[0] != self.n_chan:
              raise ValueError(f"Expected {self.n_chan} channels, got {trial_data.shape[0]}")
        if trial_data.shape[1] != self.n_samples:
              raise ValueError(f"Expected {self.n_samples} datas, got {trial_data.shape[1]}")
        if label not in ['empty', 'action']:
            raise ValueError("Label must be 'empty' or 'action'")
        trial_data = np.expand_dims(trial_data, axis=0) 
        if label == 'empty':
            self.empty_trials = np.concatenate((self.empty_trials, trial_data), axis=0)
        else:
            self.action_trials = np.concatenate((self.action_trials, trial_data), axis=0)
            
        print(f"Added trial - empty: {self.empty_trials.shape[0]}, action: {self.action_trials.shape[0]}")
      ##  print(self.empty_trials[0,:,0])
      ##  print(self.action_trials.shape)
       
       
       
       
       # self.calibration_data[label] = np.concatenate((self.calibration_data[label], trial_data), axis=0)  # Append along the first axis (n)  
                                                                          #try see np.append or np.zeros first, and then slowly change [1,:,:] [2,:,:]... and so on
        # print(trial_data)                                                #problem with np is need to know the shape first, probablly need to calculate total       
       
       
       
       # print(self.calibration_data.shape)                                      #which may leads to failure cause dataclient kinda weird, can go learn threading lock,lock after loading done,
                                                                           #and then unlock when calibration starts 

   # def Reference_Session(self,X_ref,Y_ref): #Usage: Reference_Session(load_data(session=1))
     #   self.covs_ref = self.cov_estimator.fit_transform(X_ref)
      #  self.mean_ref = mean_covariance(self.covs_ref,metric="riemann")
     #   self.covs = self.cov_estimator.fit_transform(X_ref)
      #  self.mdm.fit(self.covs, Y_ref)
     #   y_pred_mdm = self.mdm.predict(self.covs)

   
    def train(self):
        """Train the MDM classifier on collected data"""
        #if not self.calibration_data['empty'] or not self.calibration_data['action']:
          #  raise ValueError("Need both empty and action trials for training")
        
       # all_trials = self.calibration_data['empty'] + self.calibration_data['action']
        #if not all(t.shape == all_trials[0].shape for t in all_trials):
         #   raise ValueError("All trials must have the same shape")

             
        # Combine all trials
     #   print(len(self.calibration_data['empty']))
     #   print(len(self.calibration_data['empty'][0]))
        X= np.concatenate((self.empty_trials,self.action_trials),axis=0)
        print(X.shape)
        y = np.concatenate((
            np.ones(self.empty_trials.shape[0]),  # 1 for empty trials
            np.zeros(self.action_trials.shape[0]) # 0 for action trials
        ))
       # X = np.concatenate((self.calibration_data['empty'], self.calibration_data['action']), axis=0)
        # = np.array(self.calibration_data['empty'])
       # X = np.stack(self.calibration_data['empty'] + self.calibration_data['action'])   #this is just bad (old one)
       # y = np.array([1]*len(self.calibration_data['empty']) + 
      #               [0]*len(self.calibration_data['action']))
        
        # Compute covariance matrices
        covs = self.cov_estimator.fit_transform(X)
        
        # Set reference mean (from empty state)
        empty_covs = self.cov_estimator.fit_transform(self.empty_trials)
        self.mean_ref = mean_covariance(empty_covs, metric='riemann')
        
        # Train classifier
        self.mdm.fit(covs, y)
        self.is_trained = True

    def add_calibration_data_game2(self, trial_data, label):
        """Store raw EEG data for calibration"""
        if not isinstance(trial_data, np.ndarray):
              raise ValueError("trial_data must be a NumPy array")
        if trial_data.ndim != 2:
              raise ValueError("trial_data must be 2D (channels x samples)")
        if trial_data.shape[0] != self.n_chan:
              raise ValueError(f"Expected {self.n_chan} channels, got {trial_data.shape[0]}")
        if trial_data.shape[1] != self.n_samples:
              raise ValueError(f"Expected {self.n_samples} datas, got {trial_data.shape[1]}")
        if label not in ['lefthand', 'righthand']:
            raise ValueError("Label must be 'left' or 'right'")
        trial_data = np.expand_dims(trial_data, axis=0) 
        if label == 'lefthand':
            self.lefthand_trials = np.concatenate((self.lefthand_trials, trial_data), axis=0)
        else:
            self.righthand_trials = np.concatenate((self.righthand_trials, trial_data), axis=0)
            
        print(f"Added trial - left: {self.lefthand_trials.shape[0]}, right: {self.righthand_trials.shape[0]}")


        
    def train_game2(self):
        X= np.concatenate((self.lefthand_trials,self.righthand_trials),axis=0)
        print(X.shape)
        y = np.concatenate((
            np.zeros(self.lefthand_trials.shape[0]),  # 0 for left trials
            np.ones(self.righthand_trials.shape[0]) # 1 for right trials
        ))

        # use all trials as reference mean
        covs = self.cov_estimator.fit_transform(X)
        self.mean_ref = mean_covariance(covs, metric='riemann')
        
        
        # Train classifier
        self.mdm.fit(covs, y)
        accuracy = cross_val_score(self.mdm, covs, y)
        print(accuracy.mean())
        self.is_trained = True


    def add_calibration_data_game3(self, trial_data, label):
        """Store raw EEG data for calibration"""
        if not isinstance(trial_data, np.ndarray):
              raise ValueError("trial_data must be a NumPy array")
        if trial_data.ndim != 2:
              raise ValueError("trial_data must be 2D (channels x samples)")
        if trial_data.shape[0] != self.n_chan:
              raise ValueError(f"Expected {self.n_chan} channels, got {trial_data.shape[0]}")
        if trial_data.shape[1] != self.n_samples:
              raise ValueError(f"Expected {self.n_samples} datas, got {trial_data.shape[1]}")
            # Check for valid label (4-class MI)
        valid_labels = ['lefthand', 'righthand', 'feet', 'tongue']
        if label not in valid_labels:
               raise ValueError(f"Label must be one of: {valid_labels}")
        trial_data = np.expand_dims(trial_data, axis=0)  
        if label == 'lefthand':
            self.lefthand_trials = np.concatenate((self.lefthand_trials, trial_data), axis=0)
        elif label == 'righthand':
            self.righthand_trials = np.concatenate((self.righthand_trials, trial_data), axis=0)
        elif label == 'feet':
            self.feet_trials = np.concatenate((self.feet_trials, trial_data), axis=0)
        elif label == 'tongue':
            self.tongue_trials = np.concatenate((self.tongue_trials, trial_data), axis=0)
        
        print(f"Added trial - left: {self.lefthand_trials.shape[0]}, right: {self.righthand_trials.shape[0]}, feet: {self.feet_trials.shape[0]}, tongue: {self.tongue_trials.shape[0]}")
         
    def train_game3(self):
        X= np.concatenate((self.lefthand_trials,self.righthand_trials,self.feet_trials,self.tongue_trials),axis=0)
        print(X.shape)
        y = np.concatenate((
            np.zeros(self.lefthand_trials.shape[0]),  # 0 for left trials
            np.ones(self.righthand_trials.shape[0]), # 1 for right trials
        2 * np.ones(self.feet_trials.shape[0]),  # 2 for feet
        3 * np.ones(self.tongue_trials.shape[0])  # 3 for tongue
        ))

        # use all trials as reference mean
        covs = self.cov_estimator.fit_transform(X)
        self.mean_ref = mean_covariance(covs, metric='riemann')
        
        
        # Train classifier
        self.mdm.fit(covs, y)
        accuracy = cross_val_score(self.mdm, covs, y)
        print(accuracy.mean())
        self.is_trained = True

        
    def predict(self, trial_data):
        """Predict class for new trial"""
        if not self.is_trained:
            raise RuntimeError("Classifier not trained - run calibrate() first")
            
        # Compute and align covariance
       # print("test",trial_data.shape)
        trial_data = np.expand_dims(trial_data, axis=0) 
     #   print("checkpoint2",trial_data.shape)
       # cov = self.cov_estimator.transform(trial_data[np.newaxis, ...])
        cov = self.cov_estimator.transform(trial_data)
     #   print("checkpoint3",cov)
        mean_trial = mean_covariance(cov, metric='riemann')
      #  print("checkpoint4")
        R = self.mean_ref @ np.linalg.inv(mean_trial)
      #  print("checkpoint5")
        aligned_cov = R @ cov[0] @ R.T
      #  print("checkpoint6")
        print(self.mdm.predict(aligned_cov[np.newaxis, ...])[0])
        
        return self.mdm.predict(aligned_cov[np.newaxis, ...])[0]