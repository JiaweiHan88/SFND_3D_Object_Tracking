# **Track an Object in 3D Space** 


#### The goals / steps of this project are the following:
* Match 3D Objects from previous and current data frame using keypoint matches
* Compute Lidar-based TTC
* Associate Keypoint Correspondences with Bounding Boxes
* Compute Camera-based TTC
* Performance Evaluation Lidar TTC
* Performance Evaluation Camera TTC with different combination of detector/descriptor

For each rubric point a commented code snippet is presented.

[//]: # (Image References)

[image1]: ./writeup_images/lidarttc.png "nvidia"
[image2]: ./writeup_images/758.png "center"
[image3]: ./writeup_images/768.png "recovery_1"
[image4]: ./writeup_images/cameraformula.png "recovery_2"
[image5]: ./writeup_images/720.png "recovery_3"
[image6]: ./writeup_images/683.png "recovery_4"
[image7]: ./writeup_images/camerattc.png "recovery_5"
[image8]: ./writeup_images/ttcdiff.png "recovery_5"
[image9]: ./writeup_images/processing.jpg "recovery_5"

---
#### Match 3D Objects from previous and current data frame using keypoint matches

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides 
as output the ids of the matched regions of interest (i.e. the boxID property). 
Matches must be the ones with the highest number of keypoint correspondences.

Implementation can be found in `camFusion_Student.cpp` 
```sh
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //table to store number of matches between boundingboxes of the previous and current frame
    cv::Mat count_table = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32S);
    for (cv::DMatch &match_point : matches)
    {
        const auto &prev_pt = prevFrame.keypoints[match_point.queryIdx].pt;
        const auto &curr_pt = currFrame.keypoints[match_point.trainIdx].pt;
        for (size_t i = 0; i < prevFrame.boundingBoxes.size(); ++i)
        {
            for (size_t j = 0; j < currFrame.boundingBoxes.size(); ++j)
            {
                if (prevFrame.boundingBoxes[i].roi.contains(prev_pt) && currFrame.boundingBoxes[j].roi.contains(curr_pt))
                {
                    count_table.at<int>(i, j)++;
                }
            }
        }
    }
    int maxMatch, maxMatchID;
    //match according to minimum number of boundingboxes in prev and current frame, 
    //otherwise we will have duplicated matches for the frame with less bbox
    size_t minBox = min(count_table.rows, count_table.cols);
    size_t maxBox = max(count_table.rows, count_table.cols);
    bool minPrev = minBox == count_table.rows; //whether previous frame has less boundingbox
    for (size_t i = 0; i < minBox; ++i)
    {

        maxMatch = 0;
        maxMatchID = -1;
        for (size_t j = 0; j < maxBox; ++j)
        {
            if (minPrev)
            {
                if (count_table.at<int>(i, j) > maxMatch)
                {
                    maxMatchID = j;
                    maxMatch = count_table.at<int>(i, j);
                }
            }
            else
            {
                if (count_table.at<int>(j, i) > maxMatch)
                {
                    maxMatchID = j;
                    maxMatch = count_table.at<int>(j, i);
                }
            }
        }
        // if we found a best match
        if (maxMatchID != -1)
        {
            if (minPrev)
            {
                bbBestMatches.emplace(i, maxMatchID);
            }
            else
            {
                bbBestMatches.emplace(maxMatchID, i);
            }
        }
    }
}
```
#### Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. 

The implementation can be found in  `computeTTCLidar` in camFusion_Student.cpp.
I have implemented three way of calculating the previous distance and current distance. More results are presented in performance evaluation.

1. Take the minimum of each lidarpoint vector, which doesnt consider outliers.
2. Take average of the first k-th minimum x values (still affected by outlier, but reduced)
3. Use the k-th minimum value directly and skip the first few values (not affected by outlier, assumption is made that we can only have 1-3 outliers).

```
    // first sort the lidarpoints, it might be computationally more effective to only sort part of the points,
    // e.g. implement a custom bubble or quicksort algorithmus just for a percentage of the points or use a min heap.
    // here for simplicity, we just sort all points
    auto comOp = [](const LidarPoint &lp1, const LidarPoint &lp2) { return lp1.x < lp2.x; };
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), comOp);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), comOp);

    //take average of lowest 10 points
    auto sumOp = [](double sum, const LidarPoint &lp) { return sum + lp.x; };
    double prevMeanX = std::accumulate(lidarPointsPrev.begin(), lidarPointsPrev.begin() + 10, 0.0, sumOp) / 10;
    double currMeanX = std::accumulate(lidarPointsCurr.begin(), lidarPointsCurr.begin() + 10, 0.0, sumOp) / 10;

    // compute TTC in accordance with the constant velocity motion model
    double dT = 1.0 / frameRate;

    //TTC = currMeanX * dT / (prevMeanX - currMeanX); //use average over 10 minimum x values
    //TTC = lidarPointsCurr.begin()->x * dT / (lidarPointsPrev.begin()->x - lidarPointsCurr.begin()->x); //use average over 10 minimum x values
    TTC = (lidarPointsCurr.begin()+3)->x * dT / ((lidarPointsPrev.begin()+3)->x - (lidarPointsCurr.begin()+3)->x);
    cout << "Lidar TTC | " << TTC << endl;
```

#### Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

To reduce the effect of miss matches, we only use a subset of all matched points which are within the standard deviation of the mean.

```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    //create multiset with struct euclidianDistanceData (in dataStructures.h)
    //it will order all entries automatically by the euclidean distance
    auto compOP = [](const euclidianDistanceData &a, const euclidianDistanceData &b) {
        return (a.eucDist < b.eucDist);
    };
    std::multiset<euclidianDistanceData, decltype(compOP)> euclideanDistances(compOP);

    // for each match, calculate Euclidean distances between keypoints from current and previous frame
    for (const auto &match : kptMatches)
    {
        const auto &currKpt = kptsCurr[match.trainIdx];

        if (boundingBox.roi.contains(currKpt.pt))
        {
            const auto &prevKpt = kptsPrev[match.queryIdx];
            euclidianDistanceData data = {cv::norm(currKpt.pt - prevKpt.pt), currKpt, match};
            euclideanDistances.insert(data);
        }
    }
    //calculate mean and std deviation of all euclidean distances
    auto sumOp = [](double sum, const euclidianDistanceData &lp) { return sum + lp.eucDist; };
    const double euclideanDistanceMean = std::accumulate(euclideanDistances.begin(), euclideanDistances.end(), 0.0, sumOp) / euclideanDistances.size();

    const double euclideanDistanceStandardDeviation =
        std::sqrt(std::accumulate(euclideanDistances.begin(), euclideanDistances.end(), 0.0,
                                  [&euclideanDistanceMean](const double sum, const euclidianDistanceData data) 
                                  {
                                      double deviation = data.eucDist - euclideanDistanceMean;
                                      return sum + deviation * deviation;
                                  }) / euclideanDistances.size());


    //only select those which lies within the stddev boundaries from the mean, which are at least ~70% of the matches, 
    //from the output we observe that around 10% are filtered out depending on detector/descriptor
    for (auto it = euclideanDistances.begin(); it != euclideanDistances.end(); ++it)
    {
        auto distdata = (*it);
        if (fabs(distdata.eucDist - euclideanDistanceMean) < euclideanDistanceStandardDeviation)
        {
            boundingBox.keypoints.push_back((*it).keyPoint);
            boundingBox.kptMatches.push_back((*it).match);
        }
    }

    // std::cout << "[clusterKptMatchesWithROI]: points in BB "
    //           << boundingBox.boxID << " before filtering: " << euclideanDistances.size()
    //           << "; after filtering: " << boundingBox.keypoints.size()
    //           << "; Euclidean Distance Mean: " << euclideanDistanceMean
    //           << "; Euclidean distance standard deviation: " << euclideanDistanceStandardDeviation
    //           << std::endl;
}
```

#### Compute Camera-based TTC
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

The ttc is calculated using the median dist ratio in order to reduce outlier influence. The formula for the calculation is presented here:

 ![alt text][image4]
```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {

        // current keypoint and its match
        cv::KeyPoint kp1Cur = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kp1Pre = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {

            double minDist = 100.0; // min. required distance

            // next keypoint and its match
            cv::KeyPoint kp2Cur = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kp2Pre = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kp1Cur.pt - kp2Cur.pt);
            double distPrev = cv::norm(kp1Pre.pt - kp2Pre.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.empty())
    {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    size_t medidx = floor(distRatios.size() / 2.0);
    // use median dist. ratio to remove outlier influence
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medidx - 1] + distRatios[medidx]) / 2.0 : distRatios[medidx];
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}
```

#### Performance Evaluation Lidar TTC

Detailed data of the performance evaluation can be found in performance.xls

At first i was not able to find any major irregularities in the calculated data, then i changed my lidar ttc calculation to use the minimum data directly instead of making it stable against statistical outliers.
Using the previous mentioned methods, the results are presented below:

 ![alt text][image1]
 
 Using the first methode, we see at seq 12 and 17, that we get a negative ttc, meaning that the distance to the preceeding vehicle increased.
 Taking a look into the lidar points, we find the reason lies in the existing outliers, which deliver an incorrect distance, one outlier will generate two bad ttc calculations, since each dataframe is used once as current and once as previous data.
 
  ![alt text][image5]
   
  ![alt text][image6]
 
 Using the 2. and 3. methods or calculating lidar ttc, we only observe a hiccup around sequence 7, which is apparent for all three methods.
 
 Looking at the lidarpoints around seq 7, we are not observing any outliers and the distance number does confirm our calculation.
 
  ![alt text][image2]
  
  ![alt text][image3]
  
 Our approach doesnt consider the history of the data, it only looks at current and previous dataframes, we can account for outliers within a measurement, 
 but if a whole measurement is affected for some reason, our calculated ttc will get unstable.
 

###  Performance Evaluation Camera TTC with different combination of detector/descriptor

Detailed data of the performance evaluation can be found in performance.xls

The following combinations are not working:

AKAZE descriptor can only work with AKAZE keypoints, any other detector will not work with AKAZE descriptor.
SIFT keypoints can not be processed by ORB descriptor.

That leaves us 42 - 6 -1 = 35 combinations.  

The following data can be retrieved from the code by running `./3D_object_tracking  -auto`

For all combinations we observe that combinations with harris and orb detector fail catastrophically, for several frame, they dont deliver any results.
``` 
HARRIS---BRISK	
HARRIS---BRIEF	
HARRIS---ORB	
HARRIS---FREAK	
HARRIS---SIFT	
ORB---BRISK	
ORB---BRIEF	
ORB---ORB	
ORB---FREAK	
ORB---SIFT
```

For the remaining combinations we get the following results:

 ![alt text][image7] 
 
 This graph does not tell much, since we dont have a ground truth for the ttc. Using lidar calculated ttc as reference, we plot the accumulated difference over 18 sequences to find which combination
 best matches the lidar calculation.
 
  ![alt text][image8] 
  
  As can be seen in the graph, AKAZE detector based approaches have the least difference compared to lidarbased ttc. But from previous project, we know that the processing time is not optimal.
  If we consider both processing time and "accuracy".
  
  ![alt text][image9] 
  
  I would still consider FAST_BRIEF or FAST_BRIST as a valid choice
  
 