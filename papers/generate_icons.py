from PIL import Image, ImageDraw

def process_bookmark_icons(input_path):
    try:
        # Load the original image
        img = Image.open(input_path).convert("RGBA")
        width, height = img.size
        datas = img.getdata()
        
        # --- 1. Generate bookmark_star_white.png ---
        # Strategy: Keep pixels that are close to white, make everything else transparent.
        # Since the background is green (low Red/Blue) and star is white (High Red/Blue),
        # we filter by Red and Blue channels.
        
        star_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        star_data = []
        
        for item in datas:
            # item is (R, G, B, A)
            # Heuristic: White has high Red and Blue. Green has low Red and Blue.
            # Threshold: R > 150 and B > 150 catches the white star and ignores the green.
            if item[0] > 150 and item[2] > 150 and item[3] > 0:
                # Set pixel to solid White, preserving original Alpha if needed
                # (Using item[3] for alpha to keep anti-aliasing edges somewhat smooth)
                star_data.append((255, 255, 255, item[3]))
            else:
                # Make transparent
                star_data.append((0, 0, 0, 0))
        
        star_img.putdata(star_data)
        star_img.save("bookmark_star_white.png", "PNG")
        print(f"Success: Generated 'bookmark_star_white.png' ({width}x{height})")

        # --- 2. Generate bookmark_circle_mask.png ---
        # Strategy: Find the bounding box of the non-transparent area and draw a solid white circle.
        
        # Get the bounding box of the visible icon (ignores transparent padding)
        bbox = img.getbbox()
        
        if bbox:
            mask_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(mask_img)
            
            # Draw a solid white circle exactly inside the bounding box of the original shape
            draw.ellipse(bbox, fill=(255, 255, 255, 255))
            
            mask_img.save("bookmark_circle_mask.png", "PNG")
            print(f"Success: Generated 'bookmark_circle_mask.png' ({width}x{height})")
        else:
            print("Error: Could not detect content bounds for the mask.")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_path}'. Make sure the file is in the same directory.")

if __name__ == "__main__":
    # Ensure this matches your filename exactly
    process_bookmark_icons("bookmark_green.png")