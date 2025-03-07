from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import io

app = FastAPI()

# Charger le modèle une seule fois
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_depth", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

@app.post("/generate")
async def generate_image(file: UploadFile = File(...), style: str = "Moderne"):
    styles = {
        "Moderne": """
                        Transform this house into a sleek and modern home with a minimalist and sophisticated design. Incorporate clean lines, geometric shapes, and a facade made of contemporary materials such as smooth concrete, natural wood, and large glass surfaces.

                        Add floor-to-ceiling glass windows to maximize natural light and create a seamless connection between indoor and outdoor spaces. Opt for a flat or slightly sloped roof, with an extended overhang to add a bold architectural effect. Integrate spacious terraces and a patio with a stylish pergola.

                        Use a neutral and elegant color palette: white, anthracite gray, matte black, and touches of light wood to bring warmth to the design. Highlight integrated exterior lighting under the eaves and along pathways to create a sophisticated ambiance at night.

                        Include an infinity pool or a minimalist water feature with natural stone finishes to enhance the contemporary aesthetic. Finally, ensure the landscaping is well integrated, featuring a minimalist garden, lush plants, and a pathway made of polished concrete or natural stone.

                        The goal is to achieve a refined modern home that combines aesthetics, comfort, and innovative architecture.
                    """,


        "Moderne Tropicale": """
                        Transform this house into a stunning modern tropical villa, blending contemporary design with natural elements for a luxurious yet relaxing atmosphere. Emphasize open spaces, organic materials, and a seamless indoor-outdoor flow.

                        Incorporate large sliding glass doors and open-air living areas to create a fluid connection between nature and the interior. Use natural wood, exposed concrete, and stone textures to enhance the tropical aesthetic. The roof can be flat or gently sloped, possibly with overhanging eaves for shade and ventilation.

                        Add lush greenery, integrating vertical gardens, tropical plants, and palm trees around the house. Feature an infinity pool or a natural-style swimming pond, surrounded by wooden decking and an outdoor lounge area with a fire pit.

                        The facade should include earthy tones, with accents of warm wood, matte black, and deep green to blend harmoniously with the surrounding nature. Introduce pergolas, breezy terraces, and shaded outdoor lounges, maximizing comfort in a warm climate.

                        At night, incorporate soft ambient lighting, with recessed floor lights, poolside LED strips, and strategically placed lanterns to create a cozy yet elegant tropical ambiance.

                        The goal is to design a modern tropical retreat, where contemporary luxury meets nature, offering a perfect balance of sophistication and relaxation.

                        """,
        "Méditerrannéen": """
                        Transform this house into a modern Mediterranean villa, blending traditional charm with contemporary elegance. Emphasize white stucco walls, natural stone elements, and soft, organic curves that evoke the warmth and sophistication of Mediterranean architecture.

                        Use arched windows and doorways to enhance the classic aesthetic while integrating large glass openings to allow abundant natural light. The roof should feature terracotta tiles or a flat design with a rooftop terrace, ideal for enjoying panoramic views.

                        Incorporate wooden beams and pergolas, providing shade and adding texture to the exterior. Outdoor spaces should include a spacious courtyard, shaded terraces, and an infinity pool with a stone or mosaic finish, surrounded by lush Mediterranean plants such as olive trees, lavender, and bougainvillea.

                        The color palette should remain light and earthy, with soft whites, warm beige, terracotta, and accents of deep blue or sage green. The interior should flow seamlessly into the outdoor areas, with open-plan spaces, natural stone or terracotta flooring, and handcrafted details like ceramic tiles and wrought iron elements.

                        For ambiance, incorporate soft, warm lighting with lantern-style fixtures and recessed floor lighting, creating a serene and inviting Mediterranean retreat.

                        The goal is to design a timeless, sun-drenched Mediterranean home, where tradition meets modern luxury, offering comfort, elegance, and a strong connection to nature.

                    """
    } 
      
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    prompt = styles.get(style, styles["Moderne"])

    # Générer l'image stylisée
    output_image = pipeline(prompt, image=image).images[0]

    # Convertir l'image en bytes
    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return {"image_bytes": img_bytes.getvalue()}

# RunPod spécifie que l'API doit écouter sur le port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)