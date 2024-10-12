# Project description
This is taken from the Blackboard assignment description and modified for readability.

## Introduction
This is the first of your group projects.  Please work with your Pi group colleague(s) in solving this task.  Each student will have an individual file set to work with, but all students in a group are expected to develop and employ a common solution code base in generating solution files for submission to submitty.  Thus, and in general, the submitty scores from all group members will be expected to be similar.

## Retrieving files
Your individual source files are online.  You must:
1.  retrieve your file list to your Pi; (GET with a key of shortname=***yourshortusername***   e.g. shortname=bobb)
1.  retrieve your individual files to your Pi. (use a separate single get for each file in the list you downloaded in i), incorporating 2 keys: shortname as above, and myfilename=***name of each file you wish to download***)
All files are held at:   cs7ns1.scss.tcd.ie

I have asked that the cs7ns1.scss.tcd.ie file serving machine be configured to ONLY be accessible from the raspberry pi devices.  This means you *must* complete key initial elements of the task on the Pis themselves.  You should complete the maximum number of your project activities on your assigned Pi.

## Project Details

The server is configured to create a mild technical scalability challenge for you.
Each data set is unique, as is each individual solution file loaded into submitty.  If you download and work with the wrong file list and file set you will get zero.
Submitty will be available for up to 40 grading attempts for each student.  
The nature of the challenge task uses the same captcha source generator as in Project 1 so you get to build on the work you have already done. 
The symbols file and individual captcha files may differ in content to those you have seen in Project 1.   Please consider a careful pre-inspection and collaboration via Piazza to establish what you think the symbol set might be.  The symbols file is the same for all the class.  

### Recommendations
As you know, the quality, robustness and reliability of a machine learning system is significantly influenced by the quality, fidelity and integrity of the source data used.  I will not (initially) provide you with details of the font sets used.   I will not provide you with the symbols file. 
You must try to identify a reliable and trustworthy corpus of information prior to trying to solve this task.  You must also understand and work within the real-world constraints of the task. 
For example, if you are asked to build a learning or updating system to optimise network traffic, to solve captchas, to control a space craft in flight, etc the first things you must do are to understand the task, the environment, the goals, the resources -  and then seek to define, scope and bound the task towards making it achievable.
Thus a first, essential, step is to use your best collaborative efforts to identify the font, or fonts, in use and to source your own versions of these fonts.  You must also collaboratively seek to agree on the character set in use.  Once these two parameter sets are agreed, you can then proceed to build and refine your system. 
Please keep in mind that your best collaborative efforts may not result in the identification of a correct set of font sources and/or a correct charcter set.  Just because the majority may believe something does not necessarily make it correct.
I *strongly* encourage classwide participation and contribution on this e.g. via Piazza
Please note that Project 2 now incorporates some network elements that you must work with at scale.  Please remember that networks are and can be unreliable - by design or circumstance.
Again - please use Piazza for help and support!


## Resources
In the table below you will find your userid and the raspberry pi you have allocated to.  You will also find the FQDN for the Pi you will use.  You will share your Pi with other students.  
These Pis are not accessible from outside the SCSS network so you will need to VPN into them from the School VPN (if you have access) or 'jump' across by ssh from another machine within SCSS that you can already access from the outside e.g. macneill.scss.tcd.ie
I have no control over your access to any of the above - our SCSS tech support manage it - they're at help@rt.scss.tcd.ie     Please only trouble them if you are certain you have a problem - check with your collegaues etc first as many access and login impediments are easily reolved locally. 
In general we believe all necessary packages are already installed on the Pis.  All requests for additional packages must come to me for approval.  Support will not install packages for you directly.  You do not, and will not, have sudo or other privileged access to any SCSS systems or machines.  I do not usually approve new package requests as some of the missing packages are intended to create constraints that emulate those present on IoT devices.
Full details on other SCSS resources that are available to you, and on how to identify and access these resources can be found here:  https://support.scss.tcd.ie/ 
We are recommending that students use the VPN to access macneill  : https://support.scss.tcd.ie/wiki/VPN_Access .  Please be careful to enter your shortusername and (SCSS) password accurately as you will be locked out for 30 minutes after three failed attempts.


## Submittables/Gradables
For Project 2, in addition to your final submitty score(s), each Group must also submit:
1. a short 3 minute video clearly detailing the following:
    1.  What approaches did you adopt to the file access and retrieval task.  What problems did you encounter?  How did you go about addressing them?
    2. What approaches did you adopt to the captcha solving task.  What problems did you encounter?  How did you go about addressing them?
    3.  Itemize and describe, in sufficent convincing detail, the nature and detail of the activities and work you completed entirely on your Pi,
    4.  What other lessons did you learn completing Project 2 that it might benefit your colleagues to know about?
2. Complete codebase (with a full set of compile/run instructions and parameters necessary to achieve the results and performance you are reporting).
3. Summary details of your file retrieval; summary details of your preparation/pre processing; summary details on your training set creation; summary validation set details; submitty solving/solution file generation summary.  Please include timing information for all these. 
Submission tasks and deadlines are on blackboard.

## Bonus tasks
If there is sufficient demand (>10% class) we may also release preadvised bonus challenges at a predefined day and time  and accept submissions of your solutions up to time t+n mins later i.e. you will need to be ready and waiting with your solver infrastructure prepared to rapidly complete and submit the task and upload your results when they become available.  We will agree start time t   and also   test period n in class or on Piazza in advance!
Finally have (more) fun!