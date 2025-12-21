import { Request, Response } from 'express';
import { ImageService } from '../services/imageService';

export class ImageController {
    private imageService: ImageService;

    constructor() {
        this.imageService = new ImageService();
    }

    public async generateCartoonImage(req: Request, res: Response): Promise<void> {
        try {
            const { inputData } = req.body;
            const cartoonImage = await this.imageService.createCartoonImage(inputData);
            res.status(200).json({ image: cartoonImage });
        } catch (error) {
            res.status(500).json({ error: 'An error occurred while generating the cartoon image.' });
        }
    }

    public async getImage(req: Request, res: Response): Promise<void> {
        try {
            const { id } = req.params;
            const image = await this.imageService.getImageById(id);
            if (image) {
                res.status(200).json(image);
            } else {
                res.status(404).json({ error: 'Image not found.' });
            }
        } catch (error) {
            res.status(500).json({ error: 'An error occurred while retrieving the image.' });
        }
    }
}