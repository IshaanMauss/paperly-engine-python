const database = db.getSiblingDB("paperly_db");
const names = database.getCollectionNames().sort();
const output = names.map((name) => ({
  collection: name,
  count: database.getCollection(name).estimatedDocumentCount(),
}));
print(JSON.stringify(output, null, 2));
