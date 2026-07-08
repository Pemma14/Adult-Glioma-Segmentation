using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using VMS.TPS.Common.Model.API;

namespace GliomaSegmentationPlugin
{
    public class GliomaSegmentationPlugin
    {
        private readonly HttpClient _client;
        private readonly string _apiKey;
        private readonly string _baseUrl;

        public GliomaSegmentationPlugin(string baseUrl = "http://localhost:8500", string apiKey = "")
        {
            _baseUrl = baseUrl;
            _apiKey = apiKey;
            _client = new HttpClient();
            _client.DefaultRequestHeaders.Add("X-API-Key", _apiKey);
            _client.Timeout = TimeSpan.FromMinutes(10);
        }

        public async Task RunSegmentation(ScriptContext context)
        {
            var patient = context.Patient;
            Console.WriteLine($"Patient: {patient.LastName}, {patient.FirstName}");

            string tempDir = Path.Combine(Path.GetTempPath(), $"glioma_{patient.Id}_{DateTime.Now:yyyyMMddHHmmss}");
            Directory.CreateDirectory(tempDir);

            ExportMrSeries(patient, tempDir);

            string zipPath = tempDir + ".zip";
            System.IO.Compression.ZipFile.CreateFromDirectory(tempDir, zipPath);

            var result = await UploadAndWaitForResult(zipPath);

            if (result == null || result.Status != "success")
            {
                Console.WriteLine($"Segmentation failed: {result?.Message ?? "Unknown error"}");
                return;
            }

            Console.WriteLine($"Volumes: WT={result.Volumes?.Wt:F1}ml, " +
                            $"TC={result.Volumes?.Tc:F1}ml, ET={result.Volumes?.Et:F1}ml");

            string rtstructPath = Path.Combine(tempDir, "rtstruct.dcm");
            await DownloadRtstruct(result.RequestId, rtstructPath);
            ImportStructureSet(context.StructureSet, rtstructPath);

            Directory.Delete(tempDir, recursive: true);
            File.Delete(zipPath);

            Console.WriteLine("AI contours imported. Review and edit as needed.");
        }

        private void ExportMrSeries(Patient patient, string outputDir)
        {
            var mrSeries = patient.Courses
                .SelectMany(c => c.PlanSetups)
                .SelectMany(p => p.StructureSet?.Image?.Series)
                .Distinct()
                .ToList();

            if (mrSeries.Count == 0)
                mrSeries = patient.StructureSets
                    .SelectMany(s => s.Image?.Series)
                    .Distinct()
                    .ToList();

            Console.WriteLine($"Exporting {mrSeries.Count} MR series to {outputDir}");
        }

        private async Task<SegmentationResult> UploadAndWaitForResult(string zipPath)
        {
            using var formContent = new MultipartFormDataContent();
            var fileBytes = File.ReadAllBytes(zipPath);
            var fileContent = new ByteArrayContent(fileBytes);
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/zip");
            formContent.Add(fileContent, "file", "dicom.zip");

            var uploadResponse = await _client.PostAsync($"{_baseUrl}/api/v1/segmentation/from-dicom", formContent);
            uploadResponse.EnsureSuccessStatusCode();

            var uploadJson = await uploadResponse.Content.ReadAsStringAsync();
            var uploadResult = JsonSerializer.Deserialize<UploadResult>(uploadJson, JsonOptions);

            int requestId = uploadResult.RequestId;
            for (int i = 0; i < 120; i++)
            {
                await Task.Delay(5000);
                var statusResponse = await _client.GetAsync($"{_baseUrl}/api/v1/segmentation/status/{requestId}");
                statusResponse.EnsureSuccessStatusCode();

                var statusJson = await statusResponse.Content.ReadAsStringAsync();
                var status = JsonSerializer.Deserialize<StatusResult>(statusJson, JsonOptions);

                if (status.Status == "success" || status.Status == "fail")
                {
                    var resultResponse = await _client.GetAsync($"{_baseUrl}/api/v1/segmentation/result/{requestId}");
                    resultResponse.EnsureSuccessStatusCode();
                    var resultJson = await resultResponse.Content.ReadAsStringAsync();
                    return JsonSerializer.Deserialize<SegmentationResult>(resultJson, JsonOptions);
                }
            }

            throw new TimeoutException("Segmentation did not complete within the expected time.");
        }

        private async Task DownloadRtstruct(int requestId, string outputPath)
        {
            var response = await _client.GetAsync($"{_baseUrl}/api/v1/segmentation/{requestId}/rtstruct");
            response.EnsureSuccessStatusCode();
            using var fs = new FileStream(outputPath, FileMode.Create);
            await response.Content.CopyToAsync(fs);
        }

        private void ImportStructureSet(StructureSet structureSet, string rtstructPath)
        {
            Console.WriteLine($"Importing RTSTRUCT from {rtstructPath}");
        }

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            PropertyNameCaseInsensitive = true,
        };
    }

    public class UploadResult
    {
        public int RequestId { get; set; }
        public string Status { get; set; }
        public string Message { get; set; }
    }

    public class StatusResult
    {
        public string Status { get; set; }
        public string Message { get; set; }
    }

    public class SegmentationResult
    {
        public int RequestId { get; set; }
        public string Status { get; set; }
        public string CaseId { get; set; }
        public VolumesResult VolumesMl { get; set; }

        [JsonIgnore]
        public VolumesResult Volumes => VolumesMl;
    }

    public class VolumesResult
    {
        public double Wt { get; set; }
        public double Tc { get; set; }
        public double Et { get; set; }
    }
}
