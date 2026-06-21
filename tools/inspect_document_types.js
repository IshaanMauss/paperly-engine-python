const col = db.getSiblingDB("paperly_db").igcsequestions;
const types = col.aggregate([
  { $group: { _id: "$document_type", count: { $sum: 1 } } },
  { $sort: { count: -1 } },
]).toArray();

print(JSON.stringify(types, null, 2));
