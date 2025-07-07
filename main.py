from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import os
import base64
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import sobel, label
from openai import OpenAI
import re
from typing import List
import tempfile
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=openai_api_key)

# Hardcoded output path
OUTPUT_PATH = "final_output.jpg"

# Encode image to base64
def encode_image(image_data: bytes) -> str:
    """Convert image bytes to base64 string for API."""
    try:
        return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        raise ValueError(f"Error encoding image: {e}")

# Detect wall region in room image
def detect_wall_region(room: Image.Image) -> tuple:
    """Detect a wall region using edge detection and color uniformity."""
    try:
        room_gray = np.array(room.convert("L"))
        room_rgb = np.array(room.convert("RGB"))
        edges = sobel(room_gray)
        edge_magnitude = np.hypot(sobel(room_gray, axis=0), sobel(room_gray, axis=1))
        color_std = np.std(room_rgb, axis=2)
        
        threshold_edge = np.percentile(edge_magnitude, 60)
        threshold_color = np.percentile(color_std, 50)
        uniform_area = (edge_magnitude < threshold_edge) & (color_std < threshold_color)
        
        labeled, num_features = label(uniform_area)
        if num_features == 0:
            print("⚠️ No wall region detected, using full image")
            return 0, 0, room.size[0], room.size[1]
        
        sizes = [(i, (labeled == i).sum()) for i in range(1, num_features + 1)]
        largest_region = max(sizes, key=lambda x: x[1])[0]
        region = labeled == largest_region
        coords = np.where(region)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        wall_w, wall_h = x_max - x_min, y_max - y_min
        if wall_w < room.size[0] * 0.2 or wall_h < room.size[1] * 0.2:
            print("⚠️ Detected wall too small, using full image")
            return 0, 0, room.size[0], room.size[1]
        
        return x_min, y_min, wall_w, wall_h
    except Exception as e:
        print(f"❌ Error in detect_wall_region: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting wall region: {e}")

# Add frame to artwork
def add_frame(artwork: Image.Image, frame_style: str, frame_width_ratio: float = 0.05) -> Image.Image:
    """Add a frame to the artwork based on frame_style."""
    try:
        if frame_style.lower() == "no frame":
            return artwork
        
        width, height = artwork.size
        frame_width = int(min(width, height) * frame_width_ratio)
        
        new_size = (width + 2 * frame_width, height + 2 * frame_width)
        framed = Image.new("RGBA", new_size, (255, 255, 255, 0))
        
        frame_colors = {
            "wooden": (139, 69, 19, 255),
            "metal": (169, 169, 169, 255),
            "black": (0, 0, 0, 255),
            "white": (255, 255, 255, 255),
            "gold": (218, 165, 32, 255)
        }
        frame_color = frame_colors.get(frame_style.lower(), frame_colors["black"])
        
        draw = ImageDraw.Draw(framed)
        draw.rectangle((0, 0, new_size[0]-1, new_size[1]-1), fill=frame_color)
        draw.rectangle(
            (frame_width, frame_width, width + frame_width - 1, height + frame_width - 1),
            fill=(0, 0, 0, 0)
        )
        
        framed.paste(artwork, (frame_width, frame_width), mask=artwork.split()[3])
        
        shadow = Image.new("RGBA", new_size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle(
            (frame_width + 5, frame_width + 5, new_size[0], new_size[1]),
            fill=(0, 0, 0, 50)
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(5))
        final = Image.alpha_composite(shadow, framed)
        
        return final
    except Exception as e:
        print(f"❌ Error in add_frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding frame: {e}")

# Check for overlap between artworks
def check_overlap(new_x: int, new_y: int, new_w: int, new_h: int, placed_positions: List[dict]) -> bool:
    """Check if the new artwork overlaps with any previously placed artworks."""
    for pos in placed_positions:
        x, y, w, h = pos["x"], pos["y"], pos["w"], pos["h"]
        if not (new_x + new_w <= x or new_x >= x + w or new_y + new_h <= y or new_y >= y + h):
            return True
    return False

# Get positions for multiple artworks
def get_positions_from_prompt(prompt: str, room_image: bytes, artwork_images: List[bytes]) -> List[dict]:
    """Determine placement coordinates and frame styles for multiple artworks."""
    try:
        room_image_base64 = encode_image(room_image)
        artwork_images_base64 = [encode_image(img) for img in artwork_images]
        
        if not room_image_base64 or any(b64 is None for b64 in artwork_images_base64):
            print("❌ Error: Could not encode one or more images. Defaulting to center.")
            return [{"x_frac": 0.5, "y_frac": 0.5, "frame_style": "no frame"} for _ in artwork_images]
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You're an assistant that analyzes a room image and multiple artwork images to determine optimal placement coordinates and frame styles for each artwork on a wall in the room. "
                    "Respond in this format only:\n"
                    "[{ \"x_frac\": float, \"y_frac\": float, \"frame_style\": string }]\n"
                    "Where x_frac and y_frac are values between 0.0 and 1.0, representing the fractional position of each artwork's top-left corner relative to the detected wall's width and height. "
                    "frame_style is one of: 'wooden', 'metal', 'black', 'white', 'gold', or 'no frame'. "
                    "Use the room image to identify the most suitable wall (considering furniture, windows, lighting). "
                    "Use the artwork images to ensure placements suit their size, style, and colors, and prevent overlap by spacing artworks appropriately. "
                    "Parse the prompt to assign frame styles to individual artworks (e.g., 'first artwork with wooden frame, second with no frame'). "
                    "If the prompt specifies 'center of the room' or 'center of the wall,' place artworks at x_frac=0.5, y_frac=0.5 (adjusted for multiple artworks to avoid overlap). "
                    "If the prompt specifies 'left of,' place artworks with x_frac between 0.1 and 0.3. "
                    "If the prompt mentions 'increase the size of artwork,' artworks should occupy 60-70% of wall width unless a specific size is given (e.g., '25% size'). "
                    "If a specific size is mentioned (e.g., '25% size'), use that percentage of wall width. "
                    "If no frame is specified for an artwork, default to 'no frame'. "
                    "For multiple artworks, ensure non-overlapping, aesthetically balanced placements. "
                    "Examples:\n"
                    "- 'place two artworks on center of wall, first with wooden frame, second with no frame' → [{ \"x_frac\": 0.4, \"y_frac\": 0.5, \"frame_style\": \"wooden\" }, { \"x_frac\": 0.6, \"y_frac\": 0.5, \"frame_style\": \"no frame\" }]\n"
                    "- 'place artwork on left of wall with black frame, 25% size' → [{ \"x_frac\": 0.2, \"y_frac\": 0.5, \"frame_style\": \"black\" }]\n"
                    "Adjust for aesthetic balance and strictly avoid overlap."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{room_image_base64}"}},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in artwork_images_base64]
                ]
            }
        ]
        
        print(f"Sending request to OpenAI with {len(artwork_images)} artwork images")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        print(f"OpenAI raw response: {content}")
        
        # Remove Markdown code block if present
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        
        results = json.loads(content)
        
        if not isinstance(results, list) or len(results) != len(artwork_images):
            print("⚠️ Invalid response length, defaulting to center placements")
            return [{"x_frac": 0.5, "y_frac": 0.5, "frame_style": "no frame"} for _ in artwork_images]
        
        for result in results:
            if not isinstance(result.get("x_frac"), (int, float)) or not 0.0 <= result.get("x_frac") <= 1.0:
                result["x_frac"] = 0.5
            if not isinstance(result.get("y_frac"), (int, float)) or not 0.0 <= result.get("y_frac") <= 1.0:
                result["y_frac"] = 0.5
            if result.get("frame_style") not in ["wooden", "metal", "black", "white", "gold", "no frame"]:
                result["frame_style"] = "no frame"
        print(f"Parsed placements: {results}")
        return results
    except Exception as e:
        print(f"❌ GPT parse error: {e}")
        return [{"x_frac": 0.5 + i * 0.1, "y_frac": 0.5, "frame_style": "no frame"} for i in range(len(artwork_images))]

# Place multiple artworks
def place_artworks(room_image: bytes, artwork_images: List[bytes], prompt: str) -> bytes:
    """Place multiple artworks on the room with specified coordinates and frames."""
    print(f"Received prompt: {prompt}")
    print(f"Received room_image size: {len(room_image)} bytes")
    print(f"Received {len(artwork_images)} artwork images")
    try:
        room = Image.open(BytesIO(room_image)).convert("RGB")
        print(f"Room image opened: {room.size}")
    except Exception as e:
        print(f"❌ Error loading room image: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading room image: {e}")
    
    wall_x, wall_y, wall_w, wall_h = detect_wall_region(room)
    print(f"🧱 Wall detected at (x={wall_x}, y={wall_y}, w={wall_w}, h={wall_h})")
    
    size_match = re.search(r'(\d+)%\s*size', prompt, re.IGNORECASE)
    size_factor = float(size_match.group(1)) / 100 if size_match else 0.6
    min_size_factor = 0.3
    
    is_center = "center of the wall" in prompt.lower() or "center of the room" in prompt.lower()
    is_left = "left of" in prompt.lower()
    
    placements = get_positions_from_prompt(prompt, room_image, artwork_images)
    
    for idx, placement in enumerate(placements):
        if is_center:
            if len(artwork_images) == 1:
                placement["x_frac"] = 0.5
                placement["y_frac"] = 0.5
            else:
                placement["x_frac"] = 0.3 + (0.4 * idx / (len(artwork_images) - 1)) if len(artwork_images) > 1 else 0.5
                placement["y_frac"] = 0.5
        elif is_left:
            placement["x_frac"] = 0.2
            placement["y_frac"] = 0.3 + (0.4 * idx / max(1, len(artwork_images) - 1)) if len(artwork_images) > 1 else 0.5
    
    placed_positions = []
    
    for idx, (artwork_data, placement) in enumerate(zip(artwork_images, placements)):
        try:
            artwork = Image.open(BytesIO(artwork_data)).convert("RGBA")
            print(f"Artwork {idx+1} opened: {artwork.size}")
        except Exception as e:
            print(f"❌ Error loading artwork {idx+1}: {e}")
            continue
        
        frame_style = placement.get("frame_style", "no frame")
        artwork = add_frame(artwork, frame_style)
        
        current_size_factor = size_factor
        size_attempts = 0
        max_size_attempts = 3
        
        while size_attempts <= max_size_attempts:
            new_width = int(wall_w * current_size_factor)
            max_width = int(wall_w * 0.8)
            new_width = min(new_width, max_width)
            aspect_ratio = artwork.height / artwork.width
            new_height = int(new_width * aspect_ratio)
            max_height = int(wall_h * 0.8)
            if new_height > max_height:
                new_height = max_height
                new_width = int(new_height / aspect_ratio)
            
            try:
                artwork_resized = artwork.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Artwork {idx+1} resized to: ({new_width}, {new_height})")
            except Exception as e:
                print(f"❌ Error resizing artwork {idx+1}: {e}")
                break
            
            x_frac = placement.get("x_frac", 0.5)
            y_frac = placement.get("y_frac", 0.5)
            x = wall_x + int(wall_w * x_frac) - new_width // 2
            y = wall_y + int(wall_h * y_frac) - new_height // 2
            x = max(wall_x, min(x, wall_x + wall_w - new_width))
            y = max(wall_y, min(y, wall_y + wall_h - new_height))
            
            attempts = 0
            max_attempts = 5
            while check_overlap(x, y, new_width, new_height, placed_positions) and attempts < max_attempts:
                print(f"⚠️ Warning: Artwork {idx+1} overlaps, adjusting position.")
                x += int(new_width * 0.2)
                if x + new_width > wall_x + wall_w:
                    x = wall_x
                    y += int(new_height * 0.2)
                attempts += 1
            
            if attempts < max_attempts:
                room_np = np.array(room)
                artwork_area = room_np[y:y+new_height, x:x+new_width]
                if len(artwork_area) > 0 and np.std(artwork_area) > 50:
                    print(f"⚠️ Warning: Artwork {idx+1} may overlap with non-wall elements. Adjusting position.")
                    y = max(wall_y, y - 50)
                
                placed_positions.append({"x": x, "y": y, "w": new_width, "h": new_height})
                
                print(f"🖼️ Placing artwork {idx+1} at (x={x}, y={y}) with size ({new_width}x{new_height}) and {frame_style} frame")
                
                room.paste(artwork_resized, (x, y), mask=artwork_resized.split()[3])
                break
            else:
                size_attempts += 1
                if size_attempts <= max_size_attempts and current_size_factor > min_size_factor:
                    current_size_factor *= 0.8
                    print(f"⚠️ Warning: Reducing size of artwork {idx+1} to {current_size_factor*100:.0f}% of original.")
                else:
                    print(f"❌ Error: Could not find non-overlapping position for artwork {idx+1}, skipping.")
                    break
        
        if size_attempts > max_size_attempts or current_size_factor <= min_size_factor:
            continue
    
    output_buffer = BytesIO()
    room.save(output_buffer, format="JPEG", quality=95)
    return output_buffer.getvalue()

@app.get("/hello")
async def hello_world():
    return {"message": "Hello, World!"}

@app.post("/place-artworks")
async def place_artworks_endpoint(
    room_image: UploadFile = File(...),
    artwork_images: List[UploadFile] = File(...),
    prompt: str = Form(...)
):
    temp_file_path = None
    try:
        print(f"Received request with prompt: {prompt}")
        print(f"Room image filename: {room_image.filename}")
        print(f"Artwork image filenames: {[art.filename for art in artwork_images]}")
        room_data = await room_image.read()
        print(f"Room image size: {len(room_data)} bytes")
        artwork_data = [await artwork.read() for artwork in artwork_images]
        print(f"Artwork image sizes: {[len(data) for data in artwork_data]} bytes")
        
        if not room_data:
            raise HTTPException(status_code=400, detail="Room image is empty")
        if not artwork_data or any(not data for data in artwork_data):
            raise HTTPException(status_code=400, detail="One or more artwork images are empty")
        
        output_image = place_artworks(room_data, artwork_data, prompt)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(output_image)
            temp_file_path = temp_file.name
            print(f"Temporary file created at: {temp_file_path}")
        
        return FileResponse(
            temp_file_path,
            media_type="image/jpeg",
            filename=OUTPUT_PATH,
            background=BackgroundTask(os.unlink, temp_file_path)
        )
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {e}")
    finally:
        if temp_file_path:
            try:
                # Log if file still exists before response is complete
                print(f"Temporary file {temp_file_path} will be deleted by background task")
            except Exception as e:
                print(f"❌ Error logging temp file status: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)