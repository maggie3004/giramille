import { Router } from 'express';
import ImageController from '../controllers/imageController';

const router = Router();
const imageController = new ImageController();

router.post('/generate-cartoon', imageController.generateCartoonImage.bind(imageController));
router.get('/images/:id', imageController.getImage.bind(imageController));

export default router;