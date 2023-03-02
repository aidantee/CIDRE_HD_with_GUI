import spacy
from spacy.symbols import ORTH, LEMMA

nlp = spacy.load("en_core_sci_md")

nlp.tokenizer.add_special_case("+/-", [{ORTH: "+/-", LEMMA: "+/-"}])
nlp.tokenizer.add_special_case("mg.", [{ORTH: "mg.", LEMMA: "mg."}])
nlp.tokenizer.add_special_case("mg/kg", [{ORTH: "mg/kg", LEMMA: "mg/kg"}])
nlp.tokenizer.add_special_case("Gm.", [{ORTH: "Gm.", LEMMA: "Gm."}])
nlp.tokenizer.add_special_case("i.c.", [{ORTH: "i.c.", LEMMA: "i.c."}])
nlp.tokenizer.add_special_case("i.p.", [{ORTH: "i.p.", LEMMA: "i.p."}])
nlp.tokenizer.add_special_case("s.c.", [{ORTH: "s.c.", LEMMA: "s.c."}])
nlp.tokenizer.add_special_case("p.o.", [{ORTH: "p.o.", LEMMA: "p.o."}])
nlp.tokenizer.add_special_case("i.c.v.", [{ORTH: "i.c.v.", LEMMA: "i.c.v."}])
nlp.tokenizer.add_special_case("e.g.", [{ORTH: "e.g.", LEMMA: "e.g."}])
nlp.tokenizer.add_special_case("i.v.", [{ORTH: "i.v.", LEMMA: "i.v."}])
nlp.tokenizer.add_special_case("t.d.s.", [{ORTH: "t.d.s.", LEMMA: "t.d.s."}])
nlp.tokenizer.add_special_case("t.i.d.", [{ORTH: "t.i.d.", LEMMA: "t.i.d."}])
nlp.tokenizer.add_special_case("b.i.d.", [{ORTH: "b.i.d.", LEMMA: "b.i.d."}])
nlp.tokenizer.add_special_case("i.m.", [{ORTH: "i.m.", LEMMA: "i.m."}])
nlp.tokenizer.add_special_case("i.e.", [{ORTH: "i.e.", LEMMA: "i.e."}])
nlp.tokenizer.add_special_case("medications.", [{ORTH: "medications.", LEMMA: "medications."}])
nlp.tokenizer.add_special_case("mEq.", [{ORTH: "mEq.", LEMMA: "mEq."}])
nlp.tokenizer.add_special_case("a.m.", [{ORTH: "a.m.", LEMMA: "a.m."}])
nlp.tokenizer.add_special_case("p.m.", [{ORTH: "p.m.", LEMMA: "p.m."}])
nlp.tokenizer.add_special_case("M.S.", [{ORTH: "M.S.", LEMMA: "M.S."}])
nlp.tokenizer.add_special_case("ng.", [{ORTH: "ng.", LEMMA: "ng."}])
nlp.tokenizer.add_special_case("ml.", [{ORTH: "ml.", LEMMA: "ml."}])