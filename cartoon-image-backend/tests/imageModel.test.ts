import { ImageModel } from '../src/models/imageModel';

describe('ImageModel', () => {
    let imageModel: ImageModel;

    beforeEach(() => {
        imageModel = new ImageModel();
    });

    test('should generate a cartoon-style image', async () => {
        const inputParams = { /* specify parameters for cartoon image generation */ };
        const result = await imageModel.generateCartoonImage(inputParams);
        
        expect(result).toBeDefined();
        expect(result).toHaveProperty('url'); // Assuming the result has a URL property
        expect(result.url).toMatch(/\.png$|\.jpg$|\.jpeg$/); // Check if the URL ends with an image extension
    });

    test('should handle errors during cartoon image generation', async () => {
        const inputParams = { /* specify invalid parameters */ };
        
        await expect(imageModel.generateCartoonImage(inputParams)).rejects.toThrow('Error generating cartoon image');
    });
});