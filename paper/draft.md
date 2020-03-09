# Machine Learning Enhanced Clustering of Mountainbike Downhill Trails


1. Introduction
* Related Work
* Why does Classification not work properly/ What was problematic with the initial paper?
* The dataset
	* Garmin Watch
	* Pre calculated features
	* Raw acceleration data
	* 3 riders, 6 tracks
* Machine Learning enhanced clustering 
	* Speed threshold
	* Sliding windows to generate a greater number of samples
	* Feature Calculation
		* Trail rules (incline, switchbacks, ...) as max value in sub samples
			* How valuable is the switchback calculation? 
		* Classical accelerometer based features
		* Garmin pre calculated features
	* K Means Clustering
		* Experiments (Window length, sub sample length, number of clusters)
		* Plots over Map
		* Comparison to video
		* Interpretation of the clusters
		* Mapping of clusters to difficulty possible? Or can the styles be named? if so, how?
* Future Work
	*  Propper definition of "difficulty" depending on these clusters
	*  Search for trails depending on styles
	*  Archetypal analysis?
	*  Unsupervised feature engineering
	*  Which applications?

	
### Abstract
In an earlier work, we proposed a difficulty classification of mountainbike downhill trails utilizing a deep convolutional neural network. 
However, this approach is very much overfit to the riding style and subjective feeling of one rider. 
This work, follows a different, objective approach of classifying sections of trail. 
Instead of utilizing a pre-labeled dataset, we firstly calculate a set of features with a strong emphasis on the difficulty rules given by the "Singletrail Skala".
Compared to our privious work, we collect data of three different riders.
In a next step, we let an unsupervised clustering algorithm find typical styles of trail.
We then interpret the clusters, by comparing multiple samples to a synchronously recorded video of the ride.
We found the clustering pipeline to be able to properly differentiate between the styles fireroads, switchbacks, smooth and rough trail, or uphill across multiple riders.

#### UIST Edition
Using the aforementioned clustering of trail sections, we developed TANIA, a TrAil iNdIcAtor.
This device sits on the bike's handlebar and notifies the athlete 3 seconds ahead of upcoming style changes through five red LEDs.
We decoded the five styles regarding our hypothesized risk of the style with uphill sections being the least risky and rough trail being the riskiest style.
In an experiment, we mounted the device for 3 riders (TODO: that's not that many) down 3 different trails, comprising each style at least once.
We found the device to offer appropriate indications, helping the rider to be aware of upcoming style changes earlier, offering a lower perceived risk.

TrAil iNdIcAtor
TANIA

Feedback to rider:

* First risk indication?
	* Device on handlebar
	* Study with n riders on m trails?
		* Were the indications subjectively appropriate?
		* Were the indications appropriate compared to the recorded video?
		* Were the indications distracting?
		* Did the indications change your behaviour? If so, how?
		* Did you go slower because of indicated riskier sections?
		* Did you go faster because of indicated less riskier sections?
		* Would you change the order/color of indications?
		* Did you feel safer by using this device?

Time plan

* 26.02.2020 Start with devices and app
* 04.03.2020 Machine Learning Part done
* 11.03.2020 Device prototype
* 18.03.2020 Device(s) and connection to it ready (Android app?)
* 21.03.2020 + 22.03.2020 Recording
* 25.03.2020 Final recordings and questionnaires done
* 01.04.2020 Submission Deadline incl. Video

