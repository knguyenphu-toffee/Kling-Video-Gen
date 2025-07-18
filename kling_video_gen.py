import csv
import requests
import os
import time
import jwt
import random
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from openai import OpenAI
import cv2

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables only.")

# === Configuration ===
# IMPORTANT: Store these in environment variables or a secure config file!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY", "your-access-key-here")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY", "your-secret-key-here")

BASE_IMAGE_URL = "https://drive.google.com/uc?export=download&id=1UeGMHLjh6qhwfpV3tl9lA_Y4_nTRgI89"

AUDIO_DIR = "./audio_files"
VIDEO_DIR = "./final_videos"
TEMP_DIR = "./temp_kling_videos"
FRAMES_DIR = "./extracted_frames"

# Ensure output directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# === OpenAI Setup ===
client = OpenAI(api_key=OPENAI_API_KEY)

# Global variable to store the available first frame images
available_first_frames = []

def generate_kling_prompt(overlay_text, audio_script):
    """Generate a Kling video prompt using OpenAI."""
    system_prompt = """Generate one paragraph (≈60–90 words) that instructs KlingAI how to animate a still selfie into a TikTok‑ready UGC clip.

INPUTS
• Overlay_Text — static on‑screen caption
• Audio_Script — name of trending TikTok sound or spoken track (used only for timing, mood, and beat)

TASK
Write in present tense, chronological order, starting with “Subject:”.
Choreograph the performer’s visible reactions—facial expressions, head/upper‑torso moves, plus slight handheld‑camera shake and micro‑pans—so they embody the audio’s rhythm and emotion without lip‑syncing or speaking.

RULES

Action‑only performance: convey beats, drops, and spoken moments through gestures and expression, not mouth movement.

Emotions derive from the audio’s tone (playful, shocked, flirty, etc.); exaggerate just enough for phone‑size viewing.

Use authentic TikTok micro‑expressions: blinks, smirks, eyebrow pops, squints, eye‑rolls, quick head tilts/nods, shoulder pops, brief leans.

Keep it selfie‑style: performer remains fairly centered; add light camera shake, tiny re‑frames, and subtle pans—but never show or mention the phone or hands holding it.

Do NOT reference Overlay_Text, background, attire, camera gear, or invent dialogue.

Output exactly one paragraph; no lists, hashtags, or extra commentary.
"""

    user_prompt = f"""Overlay Text: {overlay_text}
    TikTok Audio Script: {audio_script}
    Please generate a Kling video prompt with a strong visual and emotional match to the above inputs."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating prompt: {e}")
        return None

def encode_jwt_token(ak, sk):
    """Generate JWT token for Kling AI authentication - EXACT method from Kling docs."""
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # Valid for 30 minutes
        "nbf": int(time.time()) - 5  # Valid from 5 seconds ago
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

def send_to_kling(prompt, image_url, duration="5"):
    """Send request to Kling API to generate video."""
    try:
        # Generate JWT token
        authorization = encode_jwt_token(KLING_ACCESS_KEY, KLING_SECRET_KEY)
        
        # Handle both string and bytes return types
        if isinstance(authorization, bytes):
            authorization = authorization.decode('utf-8')
        
        print(f"🔑 Generated API Token")
        
        # Kling API endpoint for image to video
        endpoint = "https://api-singapore.klingai.com/v1/videos/image2video"
        
        headers = {
            "Authorization": f"Bearer {authorization}",
            "Content-Type": "application/json"
        }
        
        # Payload format from Kling documentation
        payload = {
            "model_name": "kling-v1-6",
            "mode": "std",
            "duration": duration,
            "image": image_url,
            "prompt": prompt,
            "cfg_scale": 0.5
        }
        
        print(f"📡 Sending request to Kling API...")
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Request successful!")
            data = response.json()
            
            if data.get("code") == 0:
                # Extract task ID
                task_data = data.get("data", {})
                task_id = task_data.get("task_id")
                
                if task_id:
                    print(f"📋 Task created: {task_id}")
                    print("⏳ Waiting for video generation...")
                    video_url = wait_for_video_completion(task_id, authorization)
                    return video_url
                else:
                    print("❌ No task ID in response")
                    return None
            else:
                print(f"❌ API returned error code: {data.get('code')}")
                print(f"   Message: {data.get('message', 'No message')}")
                return None
                
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error sending to Kling: {e}")
        import traceback
        traceback.print_exc()
        return None

def wait_for_video_completion(task_id, api_token, max_attempts=60, delay=5):
    """Poll Kling API for video generation completion."""
    # Query endpoint for image2video tasks
    task_query_url = f"https://api-singapore.klingai.com/v1/videos/image2video/{task_id}"
    
    for attempt in range(max_attempts):
        try:
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(task_query_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("code") == 0:
                    task_data = data.get("data", {})
                    task_status = task_data.get("task_status")
                    
                    print(f"⏳ Status: {task_status} - Attempt {attempt + 1}/{max_attempts}")
                    
                    if task_status == "succeed":
                        task_result = task_data.get("task_result", {})
                        videos = task_result.get("videos", [])
                        if videos:
                            video_url = videos[0].get("url")
                            duration = videos[0].get("duration", "unknown")
                            print(f"✅ Video generated successfully!")
                            print(f"   Duration: {duration}s")
                            print(f"   URL: {video_url}")
                            return video_url
                    elif task_status == "failed":
                        error_msg = task_data.get("task_status_msg", "Unknown error")
                        print(f"❌ Video generation failed: {error_msg}")
                        return None
                        
            elif response.status_code == 404:
                print(f"⚠️ Task not found yet, waiting...")
            else:
                print(f"⚠️ Status check returned: {response.status_code}")
                
            time.sleep(delay)
            
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            time.sleep(delay)
    
    print("❌ Timeout waiting for video generation")
    return None

def download_kling_video(url, save_path):
    """Download video from Kling URL."""
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path
        else:
            raise Exception(f"Failed to download video from Kling: {r.status_code}")
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return None

def extract_frames_at_intervals(video_path, num_frames=7):
    """
    Extract frames at regular intervals starting from 1 second.
    Extracts first frame at 1s, then every 1.2 seconds after that.
    Times: 1.0s, 2.2s, 3.4s, 4.6s, 5.8s, 7.0s, 8.2s (7 frames total).
    """
    try:
        # Open video using OpenCV
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📹 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Calculate frame positions
        # First frame at 1 second, then every 1.2 seconds
        frame_times = []
        frame_times.append((0, 1.0))  # First frame at 1 second
        
        for i in range(1, num_frames):
            time_position = 1.0 + (i * 1.2)
            if time_position < duration:
                frame_times.append((i, time_position))
        
        print(f"📍 Will extract frames at times: {[t[1] for t in frame_times]}")
        
        extracted_frames = []
        
        # Extract frames by reading through the video
        current_frame = 0
        current_time = 0
        frame_index = 0
        
        while cap.isOpened() and frame_index < len(frame_times):
            ret, frame = cap.read()
            
            if not ret:
                break
                
            current_time = current_frame / fps
            
            # Check if we've reached the next extraction time
            if frame_index < len(frame_times) and current_time >= frame_times[frame_index][1]:
                i, target_time = frame_times[frame_index]
                
                # Save frame as image
                frame_filename = f"frame_{i+1}_at_{target_time:.1f}s.jpg"
                frame_path = os.path.join(FRAMES_DIR, frame_filename)
                
                # Ensure frame is valid before saving
                if frame is not None and frame.size > 0:
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        extracted_frames.append(frame_path)
                        print(f"✅ Extracted frame {i+1} at {current_time:.2f}s (target: {target_time:.1f}s)")
                    else:
                        print(f"❌ Failed to save frame {i+1}")
                else:
                    print(f"❌ Invalid frame at position {current_time:.2f}s")
                
                frame_index += 1
            
            current_frame += 1
        
        cap.release()
        
        if len(extracted_frames) < num_frames:
            print(f"⚠️ Only extracted {len(extracted_frames)} frames out of {num_frames} requested")
        
        # Verify frames are different by checking file sizes
        if extracted_frames:
            sizes = []
            for fp in extracted_frames:
                if os.path.exists(fp):
                    size = os.path.getsize(fp)
                    sizes.append(size)
                    print(f"   {os.path.basename(fp)}: {size:,} bytes")
            
            if len(set(sizes)) == 1:
                print("⚠️ WARNING: All frames have the same file size - they might be identical!")
        
        return extracted_frames
        
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        import traceback
        traceback.print_exc()
        return []

def upload_image_to_temporary_hosting(image_path):
    """
    Upload a local image to get a public URL.
    Using tmpfiles.org for temporary file hosting (no API key required).
    """
    try:
        # Get original file size for verification
        original_size = os.path.getsize(image_path)
        
        # Use tmpfiles.org - reliable and free
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('https://tmpfiles.org/api/v1/upload', files=files)
            
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                # Convert the URL to direct link format
                url = data['data']['url'].replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                print(f"✅ Uploaded: {os.path.basename(image_path)} ({original_size:,} bytes) → {url}")
                return url
        else:
            print(f"❌ Upload failed with status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error uploading {os.path.basename(image_path)}: {e}")
    
    return None

def create_base_video():
    """Create the base video with neutral pose and camera movement."""
    print("\n🎬 Creating base video with neutral pose...")
    
    base_prompt = """can you make a video where she is static for the most part but she is in a neutral position with a neutral face. Can you make this picture come to life and show her in her idle position. Can you add more camera shake. can she tilt her head to where it is more vertical upright. pan the camera to the left and then to the right to show the background of the room behind her. If there are ambient lighting LED strips in the background, change color to any color."""
    
    # Generate 10-second video
    video_url = send_to_kling(base_prompt, BASE_IMAGE_URL, duration="10")
    
    if not video_url:
        print("❌ Failed to create base video")
        return None
    
    # Download the base video
    base_video_path = os.path.join(TEMP_DIR, "base_video.mp4")
    if not download_kling_video(video_url, base_video_path):
        print("❌ Failed to download base video")
        return None
    
    print(f"✅ Base video created: {base_video_path}")
    
    # Extract 7 frames at specified intervals
    extracted_frames = extract_frames_at_intervals(base_video_path, num_frames=7)
    
    if not extracted_frames:
        print("❌ Failed to extract frames from base video")
        return None
    
    # Prepare the list of available first frames
    global available_first_frames
    available_first_frames = []  # Only use extracted frames, not the base image
    
    # Upload extracted frames and add to available frames
    for frame_path in extracted_frames:
        # Upload each frame and get a public URL
        frame_url = upload_image_to_temporary_hosting(frame_path)
        if frame_url:
            available_first_frames.append(frame_url)
        else:
            print(f"⚠️ Skipping frame due to upload failure: {frame_path}")
    
    print(f"✅ Prepared {len(available_first_frames)} first frame images from extracted frames")
    return available_first_frames

def get_random_first_frame():
    """Get a random first frame from the available options."""
    if not available_first_frames:
        print("❌ No extracted frames available!")
        return None
    
    selected_frame = random.choice(available_first_frames)
    print(f"🖼️ Selected first frame: {selected_frame}")
    return selected_frame

def overlay_text_and_audio(video_path, overlay_text, audio_path, output_path):
    """Add text overlay and audio to video."""
    try:
        video = VideoFileClip(video_path)
        duration = min(5, video.duration)

        # Load audio
        try:
            audio = AudioFileClip(audio_path).subclip(0, duration)
        except Exception as e:
            print(f"⚠️ Could not load audio: {audio_path} – {e}")
            audio = None

        # Text overlay with better error handling
        # Calculate 25% from top of video
        top_margin = int(video.h * 0.18)
    
        font_path = os.path.abspath('./tiktoksans/TiktokDisplay-Bold.ttf')
            
        # Calculate side margins (e.g., 10% of video width on each side)
        side_margin = int(video.w * 0.1)
        text_width = video.w - (2 * side_margin)
        
        # Create text clip with TikTok Sans font and transparent background
        txt_clip = TextClip(
            overlay_text,
            fontsize=48,
            color='white',
            font=font_path,  # Use specific TTF file path
            bg_color='transparent',  # Transparent background
            size=(text_width, None),  # Width with side margins
            method='caption',
            align='center',
            stroke_color='black',  # Add black stroke for better visibility
            stroke_width=2  # Stroke width for visibility
        ).set_duration(duration)
        
        # Position at 25% from top
        txt_clip = txt_clip.set_position(('center', top_margin))

        # Compose final video
        if txt_clip:
            final = CompositeVideoClip([video, txt_clip])
        else:
            final = video
            
        if audio:
            final = final.set_audio(audio)
            
        # Write video with explicit parameters
        final.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac", 
            fps=30,
            temp_audiofile='temp-audio.m4a',  # Specify temp audio file
            remove_temp=True  # Clean up temp files
        )
        
        # Clean up
        video.close()
        if audio:
            audio.close()
        if txt_clip:
            txt_clip.close()
        final.close()
        
        return True
    except Exception as e:
        print(f"❌ Error creating final video: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_csv(csv_path):
    """Process CSV file to generate videos."""
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # First, create the base video and extract frames
    if not create_base_video():
        print("❌ Failed to create base video. Exiting.")
        return
        
    successful = 0
    failed = 0
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        # Assume CSV has no header and columns are: Overlay, AudioScript, AudioFile
        reader = csv.reader(csvfile)
        
        for idx, row in enumerate(reader, 1):
            try:
                # Check if row has the expected 3 columns
                if len(row) < 3:
                    print(f"❌ Row {idx}: Invalid format (expected 3 columns, got {len(row)})")
                    failed += 1
                    continue
                
                overlay = row[0].strip()
                script = row[1].strip()
                audio_file = row[2].strip()
                
                if not all([overlay, script, audio_file]):
                    print(f"❌ Row {idx}: Missing required data")
                    failed += 1
                    continue
                    
                audio_path = os.path.join(AUDIO_DIR, audio_file)

                print(f"\n🎬 Processing row {idx}...")
                print(f"   Overlay: {overlay}")
                print(f"   Audio Script: {script}")
                print(f"   Audio File: {audio_file}")

                if not os.path.exists(audio_path):
                    print(f"❌ Missing audio file: {audio_file}")
                    failed += 1
                    continue

                # Get a random first frame for this video
                first_frame_url = get_random_first_frame()

                # Generate Kling prompt
                kling_prompt = generate_kling_prompt(overlay, script)
                if not kling_prompt:
                    print("❌ Failed to generate Kling prompt")
                    failed += 1
                    continue
                    
                print(f"📜 Kling Prompt Created")

                # Send to Kling API with the selected first frame
                video_url = send_to_kling(kling_prompt, first_frame_url)
                if not video_url:
                    print("❌ Skipping row due to Kling video failure.")
                    failed += 1
                    continue

                # Download video
                temp_video_path = os.path.join(TEMP_DIR, f"kling_temp_{idx}.mp4")
                if not download_kling_video(video_url, temp_video_path):
                    print("❌ Failed to download video")
                    failed += 1
                    continue
                    
                print(f"📥 Downloaded Kling video to: {temp_video_path}")

                # Create final video
                output_path = os.path.join(VIDEO_DIR, f"final_video_{idx}.mp4")
                if overlay_text_and_audio(temp_video_path, overlay, audio_path, output_path):
                    print(f"✅ Final video saved: {output_path}")
                    successful += 1
                else:
                    print(f"❌ Failed to create final video")
                    failed += 1
                    
                # Clean up temp file
                try:
                    os.remove(temp_video_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                failed += 1
    
    print(f"\n📊 Summary: {successful} successful, {failed} failed")

# === Run with Timer ===
if __name__ == "__main__":
    # Check for API keys
    if OPENAI_API_KEY == "your-api-key-here":
        print("❌ Please set your OPENAI_API_KEY environment variable")
        exit(1)
    if KLING_ACCESS_KEY == "your-access-key-here":
        print("❌ Please set your KLING_ACCESS_KEY environment variable")
        exit(1)
    if KLING_SECRET_KEY == "your-secret-key-here":
        print("❌ Please set your KLING_SECRET_KEY environment variable")
        exit(1)
    
    print("🔐 API Keys loaded:")
    print(f"   OpenAI: {OPENAI_API_KEY[:20]}...{OPENAI_API_KEY[-4:]}")
    print(f"   Kling Access: {KLING_ACCESS_KEY[:10]}...{KLING_ACCESS_KEY[-4:]}")
    print(f"   Kling Secret: {'*' * 10}...{'*' * 4}")
    print()
        
    start_time = time.time()
    
    csv_file = "videos.csv"
    print(f"📁 Processing CSV file: {csv_file}")
    process_csv(csv_file)
    
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print(f"\n⏱️ All tasks completed in {duration_minutes:.2f} minutes.")