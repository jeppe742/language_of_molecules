import sqlite3
class Database():
    def __init__(self,db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)

    def setup(self):
        """Creates the tables in the database"""

        self.conn.executescript("""
        
        CREATE TABLE IF NOT EXISTS molecules(
            molecule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            length INT,
            smiles TEXT
        );

        
        CREATE TABLE IF NOT EXISTS models(
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_name_short TEXT
        );

        CREATE TABLE IF NOT EXISTS samples(
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            idx INT,
            charge INT,
            num_neighbours INT,
            num_masks INT,
            molecule_id INT,

            FOREIGN KEY (molecule_id)
                REFERENCES molecules (molecule_id) 

        );

        
        CREATE TABLE IF NOT EXISTS 
        predictions(
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction INT,
            target INT,
            cross_entropy NUMERIC,
            num_masks INT,
            charge INT,
            num_neighbours INT,


            model_id INT,
            molecule_id INT,

            FOREIGN KEY (model_id)
                REFERENCES models (model_id),

            FOREIGN KEY (molecule_id)
                REFERENCES molecules (molecule_id)

        );
        DROP TABLE IF EXISTS atoms;
        CREATE TABLE atoms(
            atom_id INT,
            atom TEXT
        );
        INSERT INTO atoms
        VALUES (0,'H'),(1,'C'),(2,'O'),(3,'N'),(4,'F'),(5,'P'),(6,'S'),(7,'Cl'),(8,'Br'),(9,'I')
        ;

        CREATE INDEX IF NOT EXISTS idx_molecule on molecules(molecule_id, smiles);
        CREATE INDEX IF NOT EXISTS idx_sample on samples(molecule_id,idx,num_masks);
            
            """)
        #  --CREATE INDEX idx_predictions
        # --ON predictions(model_id,sample_id);

        # --CREATE INDEX idx_samples
        # --on samples(num_masks);
        
        


    def stage_results(self, insert_values):
        self.conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS 
        data(
            length INT, 
            model_name TEXT,
            model_name_short TEXT,
            num_masks INT, 
            prediction INT, 
            target INT, 
            cross_entropy NUMERIC,
            smiles TEXT,
            charge INT,
            num_neighbours INT
        );""")
        self.conn.executemany("""
            INSERT INTO 
            temp.data (length,
                        model_name, 
                        model_name_short,
                        num_masks, 
                        prediction, 
                        target, 
                        cross_entropy, 
                        smiles,
                        charge,
                        num_neighbours
                        )
            VALUES(?,?,?,?,?,?,?,?,?,?)""", insert_values)

    def apply_staged(self):


        self.conn.executescript("""
        
        INSERT INTO models(model_name, model_name_short)
        SELECT distinct model_name, model_name_short from temp.data where model_name not in (select distinct model_name from models);
        """)

        self.conn.executescript("""
        INSERT INTO molecules(smiles, length)
        SELECT distinct smiles, length from temp.data where smiles not in (select distinct smiles from molecules);

        """)
        # print("samples")
        # self.conn.executescript("""
        # INSERT INTO samples(idx, charge, num_neighbours, num_masks, molecule_id)
        # SELECT distinct temp.data.idx, temp.data.charge, temp.data.num_neighbours, temp.data.num_masks, molecules.molecule_id 
        # FROM temp.data
        # JOIN molecules on molecules.smiles = temp.data.smiles
        # LEFT JOIN samples on samples.idx=temp.data.idx
        #     AND samples.num_masks=temp.data.num_masks 
        #     AND samples.molecule_id=molecules.molecule_id

        # where samples.sample_id is null
        # ;
        # """)
# WHERE NOT EXISTS( 
#             SELECT 1 
#             FROM samples 
#             JOIN molecules on molecules.molecule_id=samples.molecule_id 
#             WHERE 
#             samples.idx=temp.data.idx
#             AND samples.num_masks=temp.data.num_masks 
#             AND molecules.smiles=temp.data.smiles  
#             )
        # print("predictions")
        self.conn.executescript("""
        INSERT INTO 
        predictions(prediction,
                    target,
                    cross_entropy,
                    num_masks,
                    charge,
                    num_neighbours,
                    model_id,
                    molecule_id
        )
        SELECT  prediction,
                target,
                cross_entropy,
                num_masks,
                charge,
                num_neighbours,
                model_id,
                molecule_id
        from temp.data 
        join models on models.model_name=temp.data.model_name
        join molecules on molecules.smiles=temp.data.smiles
 
        ;

        """)

        self.conn.executescript("""

        drop table temp.data;
        """)
        self.conn.commit()
