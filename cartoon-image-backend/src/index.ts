import express from 'express';
import { json } from 'body-parser';
import imageRoutes from './routes/imageRoutes';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(json());
app.use('/api/images', imageRoutes);

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});