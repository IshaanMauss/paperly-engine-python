const fs = require("fs");

const database = db.getSiblingDB("paperly_db");
const qpCol = database.igcsequestions;
const msCol = database.igcsemarkingschemes;

const qp = qpCol.aggregate([
  { $match: { document_type: "Question Paper" } },
  { $sample: { size: 500 } },
]).toArray();

const ms = msCol.aggregate([
  { $sample: { size: 500 } },
]).toArray();

fs.writeFileSync("sample_500_qp.json", JSON.stringify(qp, null, 2));
fs.writeFileSync("sample_500_ms.json", JSON.stringify(ms, null, 2));

print(JSON.stringify({
  qp: qp.length,
  ms: ms.length,
  files: ["sample_500_qp.json", "sample_500_ms.json"],
}));
