
# Getting Started to construct VideoQA datasets

Before starting to construct VideoQA datasets, we recommend maintaining the following file structure:
```
project/
├── code/
└── processed_data/
    ├── origin_frames/
    ├── split_frames/
    ├── split_videos/
    ├── merge_videos/
    └── meta/
	├── error/
	├── merge/
	├── QApairs/
	├── split_times/
        ├── split_audio/
        ├── split_boxes/
        ├── split_action/
        ├── split_event/
        ├── split_object/
        └── split_reasoning/
        
```

---


Here's the sequence of steps for the entire construction process:

## Splitting original videos
Step 1: First, you need to extract frames from your original videos:
```
python code/tools/frames/moviepy_frames_get.py --input_dir {origin_video_path} --save_dir processed_data/origin_frames
```
Step 2: Use the CLIP model to analyze video frames and determine splitting points (Note: Make sure you have set up an environment that can use the CLIP model):
```
python code/tools/clip/clip_video_splitting.py --input_dir processed_data/origin_frames --save_path processed_data/meta/splitting_points.json
```
Step 3: Use the splitting point information to segment the original videos and corresponding frame files into multiple smaller video clips:
```
python code/tools/clip/splitting_move.py --input_frames_path processed_data/origin_frames --input_videos_path {origin_video_path}  --input_splitting_points processed_data/meta/splitting_points.json --output_frames_path processed_data/split_frames --output_videos_path processed_data/split_videos
```

## Audio information extraction
Before extracting audio information from videos, ensure that your computer is configured to use the WhisperX model. Download it from [WhisperX](https://github.com/m-bain/whisperX)

Step 1: Start extracting audio from the split video clips and save as JSON files:
```
python code/tools/audio/whisperx_get_audio.py --input_videos_path processed_data/split_videos --save_path processed_data/meta/split_audio/audio_origin.json
```
Step 2: Modify audio information:
```
python code/tools/audio/audio_change.py --audio_origin processed_data/meta/split_audio/audio_origin.json --audio_save processed_data/meta/split_audio/audio_change.json
```

## Object boundary extraction in videos
Install GSAM from [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). You may need to manually change the model path in _automatic_label_sam_videos.py_.

Step 1: Extract object bounding boxes from video frames and retain original information:
```
python code/tools/boxes/automatic_label_sam_videos.py --input_image_dir processed_data/split_frames --output_dir processed_data/meta/split_boxes/boxes
```

Step 2: Change the format of the original information and save the output as a JSON file:
```
python code/tools/boxes/boxes_change.py --boxes_origin processed_data/meta/split_boxes/boxes --boxes_save processed_data/meta/split_boxes/video_boxes.json
```

## Video clip Annotation
The following code annotates each video clip, iterating through all frames of each video clip stored in processed_data/split_frames. In each annotation, we introduce the auxiliary information obtained above to help the large model understand the video stream, improving the annotation quality from multiple aspects.
Note: If a video's annotation file is repeatedly deleted, please check the corresponding video file and error messages.
The following annotation processes are similar, but event annotation should be run after action annotation is complete.

### Action annotation
Step 1: The following command annotates actions in the video. The parameters video_titles and video_categorys are JSON file paths recording the titles and categories of the original videos:
```shell
python code/tools/gpt/annotate/action_annotate.py --audios processed_data/meta/split_audio/audio_change.json --frames processed_data/split_frames --save processed_data/meta/split_action/action --error_log processed_data/meta/error/error_annotate_action.json --video_titles processed_data/meta/{id2titles}.json --video_categorys processed_data/meta/{categorys}.json
```
Step 2: Process and merge annotation information, while checking for video IDs with output format errors. Note: When GPT identifies a video as violating standards, the code will not record it as an error video ID, as this is unavoidable. We can filter out and specially process these videos later:
```shell
python code/tools/gpt/post_process/post_process_action.py --input processed_data/meta/split_action/action --save processed_data/meta/split_action/id2action.json --error_save processed_data/meta/error/error_post_action.json
```
Step 3: Based on the error type, delete the annotation information of error video IDs.**If successfully deleted, you need to return to step 1 to re-run the annotation code:**
```shell
python code/tools/delete_files.py --input processed_data/meta/split_action/action --error_file processed_data/meta/error/error_post_action.json
```

### Object annotation
Similar to action annotation, follow these three steps:

Step 1:
```shell
python code/tools/gpt/annotate/object_annotate.py --audios processed_data/meta/split_audio/audio_change.json --frames processed_data/split_frames --save processed_data/meta/split_object/object --boxes processed_data/meta/split_boxes/video_boxes.json --video_titles processed_data/meta/panda_2m_validation_id2title.json --video_categorys processed_data/meta/panda_2m_validation_id2category.json
```
Step 2:  
```shell
python code/tools/gpt/post_process/post_process_object.py --input processed_data/meta/split_object/object --save processed_data/meta/split_object/id2object.json --error_save processed_data/meta/error/error_post_object.json
```
Step 3:
```shell
python code/tools/delete_files.py --input processed_data/meta/split_object/object --error_file processed_data/meta/error/error_post_object.json
```

### Reasoning annotation
Step 1:
```shell
python code/tools/gpt/annotate/reasoning_annotate.py --audios processed_data/meta/split_audio/audio_change.json --frames processed_data/split_frames --save processed_data/meta/split_reasoning/reasoning --video_titles processed_data/meta/panda_2m_validation_id2title.json --video_categorys processed_data/meta/panda_2m_validation_id2category.json
```
Step 2:  
```shell
python code/tools/gpt/post_process/post_process_reasoning.py --input processed_data/meta/split_reasoning/reasoning --save processed_data/meta/split_reasoning/id2reasoning.json --error_save processed_data/meta/error/error_reasoning.json
```
Step 3:
```shell
python code/tools/delete_files.py --input processed_data/meta/split_reasoning/reasoning --error_file processed_data/meta/error/error_reasoning.json
```

### Event annotation
Step 1: Event annotation requires action annotation information, so it should be completed after action annotation
```shell
python code/tools/gpt/annotate/event_annotate.py --audios processed_data/meta/split_audio/audio_change.json --frames processed_data/split_frames --save processed_data/meta/split_event/event --boxes processed_data/meta/split_boxes/video_boxes.json --action processed_data/meta/split_action/action --error_log processed_data/meta/error/error_annotate_event.json --video_titles processed_data/meta/{id2titles}.json --video_categorys processed_data/meta/{categorys}.json
```
Step 2:  
```shell
python code/tools/gpt/post_process/post_process_event.py --input processed_data/meta/split_event/event --save processed_data/meta/split_event/id2event.json --error_save processed_data/meta/error/error_event.json
```
Step 3:
```shell
python code/tools/delete_files.py --input processed_data/meta/split_event/event --error_file processed_data/meta/error/error_event.json
```

After running the above steps, you have completed the first major part of constructing the VideoQA dataset: you have extracted multiple information from the split videos and completed various annotations, preparing for the next step of merging videos and constructing QA pairs.

## Merging videos from multiple time segments
We can flexibly merge these video clips (from the same original video) based on the desired video duration. Here's the process for completing the merge:
Step 1: Get the specific duration of each video clip
```shell
python code/tools/merge/get_id2time.py --input_videos_path processed_data/split_videos --save_path processed_data/meta/split_times/video_id2time.json
```
Step 2: Obtain video merging information for different duration ranges and save it to the corresponding JSON files.
For example, after running the code below, the file video_merge_info_60t120.json will store information about merged videos with durations between 60s-120s:
```shell
python code/tools/merge/video_merge.py --id2time_path processed_data/meta/split_times/video_id2time.json --input_videos_path processed_data/split_videos --save_path processed_data/meta/merge --save_name video_merge_info
```

After obtaining merging information for different duration ranges, you can start the following process based on the desired merged video duration range. Below, I'll demonstrate the process for merged videos with durations between 120s-300s.

## Constructing QA pairs for 120s-300s merged videos
Step 1: Merge previous video clips according to the merging information to obtain corresponding video files:
```shell
python code/tools/merge/merge_videos.py --videos processed_data/split_videos --timestamp processed_data/meta/merge/video_merge_info_120t300.json --save processed_data/merge_videos/120t300
```
Step 2: Merge basic information of video clips for different duration ranges, including action, audio, event, and object information
Merge action:
```shell
python code/tools/merge/merge_action.py --timestamp processed_data/meta/merge/video_merge_info_120t300.json --id2time processed_data/meta/split_times/video_id2time.json --action processed_data/meta/split_action/id2action.json --frames processed_data/split_frames --save processed_data/meta/merge/merge_action_120t300.json
```
Merge audio:
```shell
python code/tools/merge/merge_audio.py --timestamp processed_data/meta/merge/video_merge_info_120t300.json --id2time processed_data/meta/split_times/video_id2time.json --audio processed_data/meta/split_audio/audio_origin.json --frames processed_data/split_frames --save processed_data/meta/merge/merge_audio_120t300.json
```
Merge event:
```shell
python code/tools/merge/merge_event.py --timestamp processed_data/meta/merge/video_merge_info_120t300.json --id2time processed_data/meta/split_times/video_id2time.json --event processed_data/meta/split_event/id2event.json --frames processed_data/split_frames --save processed_data/meta/merge/merge_event_120t300.json
```
Merge object:
```shell
python code/tools/merge/merge_object.py --timestamp processed_data/meta/merge/video_merge_info_120t300.json --id2time processed_data/meta/split_times/video_id2time.json --object processed_data/meta/split_object/id2object.json --frames processed_data/split_frames --save processed_data/meta/merge/merge_object_120t300.json
```

We now have multiple merged videos and their corresponding basic information. Next, we'll use a large language model to construct the final video QA pairs.
Note that in the following code, different types of construction (action, event, object) can be run simultaneously.

### Constructing action QA pairs
Step 1: Generate QA pairs:
```shell
python code/tools/gpt/merge_annotate/event.py --time 120t300 --events processed_data/meta/merge/merge_event_120t300.json --audios processed_data/meta/merge/merge_audio_120t300.json --save processed_data/meta/merge/reannotate_events/120t300 --video_titles processed_data/meta/panda_2m_validation_id2title.json --video_categorys processed_data/meta/panda_2m_validation_id2category.json
```
Step 2: Organize and identify QA pairs, merge output into the final JSON file. During the merging process, it will check for video IDs with output format errors:
```shell
python code/tools/gpt/merge_post_process/post_process_merge.py --input processed_data/meta/merge/reannotate_actions/120t300 --save processed_data/meta/QApairs/120t300/120t300_actions_QA.json --error_save processed_data/meta/error/120t300/error_merge_action_120t300.json
```
Step 3: Delete information with output format errors. **If successfully deleted here, you should return to step 1 to re-run the QA construction code.**
```shell
python code/tools/delete_files.py --input processed_data/meta/merge/reannotate_actions/120t300 --error_file processed_data/meta/error/120t300/error_merge_action_120t300.json
```

### Constructing event QA pairs
Step 1:
```shell
python code/tools/gpt/merge_annotate/event.py --time 120t300 --events processed_data/meta/merge/merge_event_120t300.json --audios processed_data/meta/merge/merge_audio_120t300.json --save processed_data/meta/merge/reannotate_events/120t300 --video_titles processed_data/meta/panda_2m_validation_id2title.json --video_categorys processed_data/meta/panda_2m_validation_id2category.json
```

Step 2:
```shell
python code/tools/gpt/merge_post_process/post_process_merge.py --input processed_data/meta/merge/reannotate_events/120t300 --save processed_data/meta/QApairs/120t300/120t300_events_QA.json --error_save processed_data/meta/error/120t300/error_merge_events_120t300.json
```

Step 3:
```shell
python code/tools/delete_files.py --input processed_data/meta/merge/reannotate_events/120t300 --error_file processed_data/meta/error/120t300/error_merge_events_120t300.json
```

### Constructing object QA pairs
Step 1:
```shell
python code/tools/gpt/merge_annotate/object.py --time 120t300 --events processed_data/meta/merge/merge_event_120t300.json --objects processed_data/meta/merge/merge_object_120t300.json --save processed_data/meta/merge/reannotate_objects/120t300 --video_titles processed_data/meta/panda_2m_validation_id2title.json --video_categorys processed_data/meta/panda_2m_validation_id2category.json
```

Step 2:
```shell
python code/tools/gpt/merge_post_process/post_process_merge.py --input processed_data/meta/merge/reannotate_objects/120t300 --save processed_data/meta/QApairs/120t300/120t300_objects_QA.json --error_save processed_data/meta/error/120t300/error_merge_objects_120t300.json
```

Step 3:
```shell
python code/tools/delete_files.py --input processed_data/meta/merge/reannotate_objects/120t300 --error_file processed_data/meta/error/120t300/error_merge_objects_120t300.json
```

The above demonstrates the QA construction process for videos in the 120s-300s range. If you need other time ranges, check the merging information we obtained and modify the time range in the above process according to your desired time range. Remember to follow the conventions used above when modifying the time range.

---

At this point, you have completed the construction of QA pairs for merged videos. The final storage path for the QA pairs is processed_data/meta/QApairs. You can check the finally generated QA pairs at this location.
