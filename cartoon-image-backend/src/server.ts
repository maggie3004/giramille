import express from 'express';
import bodyParser from 'body-parser';
import imageRoutes from './routes/imageRoutes';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use('/api/images', imageRoutes);

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});