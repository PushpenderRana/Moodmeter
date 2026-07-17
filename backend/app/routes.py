from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import tempfile

from app.schemas import ReviewRequest
from app.predictor import predict_sentiment
from app.batch_processor import process_csv

router = APIRouter()

@router.post("/predict")
def predict(data: ReviewRequest):

    result = predict_sentiment(data.review)

    return {
        "review": data.review,
        **result
    }


@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    result_df = process_csv(temp_path)

    output_file = "prediction_results.csv"

    result_df.to_csv(output_file, index=False)

    return FileResponse(
        output_file,
        media_type="text/csv",
        filename="prediction_results.csv"
    )